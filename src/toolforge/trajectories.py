"""Trajectory records: the persisted form of one verified (or rejected) episode.

A trajectory contains everything needed to (a) audit what the model saw and
did, (b) rebuild SFT samples with the exact training/inference serialization,
and (c) recompute rewards — request, model-visible tool schemas and system
prompt, every assistant action, every tool observation, the final response,
the verifier verdict, and reward components when computed.

Source labels are kept on every record: ``reference`` (spec-compiled
deterministic solutions), ``self`` (rollouts of the model being trained),
``teacher`` (optional external API model), ``random`` (baseline). SFT/GRPO
mixing filters on these labels and on the split — test-split trajectories are
refused at construction time.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolforge.executor import EpisodeResult
from toolforge.schemas import GRAMMAR_VERSION, ToolForgeError
from toolforge.tasks import Task
from toolforge.verifier import VerificationResult

SOURCES = ("reference", "self", "teacher", "random")


class TrajectoryError(ToolForgeError):
    """Raised for malformed trajectory records or split violations."""


def messages_from_episode(task: Task, episode: EpisodeResult) -> list[dict[str, str]]:
    """Reconstruct the exact model-visible conversation of an episode."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": episode.system_prompt},
        {"role": "user", "content": task.request},
    ]
    for step in episode.trace:
        messages.append({"role": "assistant", "content": step.raw_output})
        if step.tool_result is not None:
            messages.append({"role": "tool",
                             "content": json.dumps(step.tool_result, ensure_ascii=False)})
    return messages


@dataclass
class Trajectory:
    """One collected episode with its verdict (and optional reward)."""

    trajectory_id: str
    task_id: str
    split: str
    family: str
    category: str
    difficulty: str
    source: str
    policy_id: str
    sampling_seed: int
    request: str
    messages: list[dict[str, str]]
    actions: list[dict[str, Any]]
    termination: str
    final_output: str | None
    verified: bool
    verification: dict[str, Any]
    reward: dict[str, Any] | None = None
    tool_call_count: int = 0
    invalid_call_count: int = 0
    parse_error_count: int = 0
    grammar_version: str = GRAMMAR_VERSION
    tools_fingerprint: str = ""
    benchmark_seed: int = 0
    benchmark_version: str = "v1"
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise TrajectoryError(f"Unknown trajectory source {self.source!r} (allowed: {SOURCES})")
        if self.split == "test":
            raise TrajectoryError(
                f"Refusing to build a training trajectory for test-split task {self.task_id}; "
                "the test split is never used for data collection or tuning")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        known = cls.__dataclass_fields__
        unknown = set(data) - set(known)
        if unknown:
            raise TrajectoryError(f"Unknown trajectory field(s): {sorted(unknown)}")
        missing = {"trajectory_id", "task_id", "split", "family", "category", "source",
                   "messages", "verified", "verification"} - set(data)
        if missing:
            raise TrajectoryError(f"Trajectory record missing field(s): {sorted(missing)}")
        return cls(**data)


def trajectory_id(task_id: str, source: str, run_tag: str, index: int) -> str:
    """Stable, collision-resistant id: task + source + short content hash."""
    digest = hashlib.sha256(f"{task_id}|{source}|{run_tag}|{index}".encode()).hexdigest()[:8]
    return f"traj-{task_id}-{source}-{digest}"


def trajectory_from_episode(task: Task, episode: EpisodeResult, verdict: VerificationResult, *,
                            source: str, policy_id: str, sampling_seed: int = 0,
                            reward: dict[str, Any] | None = None,
                            tools_fingerprint: str = "", run_tag: str = "",
                            index: int = 0, extra: dict[str, Any] | None = None) -> Trajectory:
    actions = [step.action for step in episode.trace if step.action is not None]
    return Trajectory(
        trajectory_id=trajectory_id(task.task_id, source, run_tag, index),
        task_id=task.task_id, split=task.split, family=task.family, category=task.category,
        difficulty=task.difficulty, source=source, policy_id=policy_id,
        sampling_seed=sampling_seed, request=task.request,
        messages=messages_from_episode(task, episode), actions=actions,
        termination=episode.termination, final_output=episode.final_output,
        verified=verdict.success, verification=verdict.to_dict(), reward=reward,
        tool_call_count=episode.tool_call_count, invalid_call_count=episode.invalid_call_count,
        parse_error_count=episode.parse_error_count,
        tools_fingerprint=tools_fingerprint,
        benchmark_seed=task.benchmark_seed, benchmark_version=task.benchmark_version,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extra=extra or {},
    )


def save_trajectories(path: str | Path, trajectories: list[Trajectory]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(json.dumps(traj.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(trajectories)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    path = Path(path)
    if not path.exists():
        raise TrajectoryError(f"Trajectory file not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    out.append(Trajectory.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TrajectoryError, TypeError) as exc:
                    raise TrajectoryError(f"{path}:{line_no}: {exc}") from exc
    return out
