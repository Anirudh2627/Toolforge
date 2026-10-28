"""Trajectory collection: reference solutions, model rollouts, rejection sampling.

Three data sources (spec §7), all verifier-gated:

1. **Reference trajectories** — compiled from expected outcomes by
   :mod:`toolforge.reference_solutions`, then *executed and verified*; only
   verified episodes are kept, and coverage is reported honestly per category.
2. **Self-generated trajectories** — rollouts of the current open model with
   rejection sampling: keep verified successes, record rejection rates and
   failure categories. Never assumes the model can succeed unaided; reference
   data is the working fallback.
3. **Teacher trajectories** (optional) — any policy (e.g.
   :class:`toolforge.inference.OpenAICompatiblePolicy`) goes through the same
   collection path; API keys are read from the environment only.

Split hygiene is enforced here, not left to convention: collecting on
``test``-split tasks raises immediately, and :func:`build_sft_mix` refuses
test-split trajectories and held-out families.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from toolforge import get_logger
from toolforge.executor import EpisodeResult, Executor, PolicyContext
from toolforge.reference_solutions import compile_reference_solution, last_compilation_failure_reason
from toolforge.rewards import RewardConfig, compute_reward
from toolforge.tasks import Task
from toolforge.trajectories import Trajectory, save_trajectories, trajectory_from_episode
from toolforge.verifier import verify_episode

logger = get_logger("data_generation")


class DataGenerationError(Exception):
    """Raised on split violations or unusable collection inputs."""


# --------------------------------------------------------------------------
# Scripted policy with runtime placeholders
# --------------------------------------------------------------------------

def resolve_prev(value: Any, outputs: list[Any]) -> Any:
    """Recursively replace ``{"$prev": "path.to.field"}`` from recent outputs.

    Resolution scans tool outputs newest-first; the first output containing the
    full path wins. A missing path raises ``KeyError`` — a broken reference
    script must fail the episode visibly, not fabricate an id.
    """
    if isinstance(value, dict):
        if set(value) == {"$prev"}:
            path = value["$prev"].split(".")
            for output in reversed(outputs):
                cursor: Any = output
                try:
                    for part in path:
                        cursor = cursor[part]
                    return copy.deepcopy(cursor)
                except (KeyError, TypeError, IndexError):
                    continue
            raise KeyError(f"cannot resolve $prev path {value['$prev']!r} in recent outputs")
        return {k: resolve_prev(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_prev(v, outputs) for v in value]
    return value


class SolutionPolicy:
    """Replays a compiled reference script, resolving ``$prev`` placeholders.

    Unlike :class:`toolforge.executor.ScriptedPolicy`, this policy reads the
    conversation's tool observations, so multi-step flows (draft -> send,
    create -> update) work with runtime-generated ids.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self._index = 0
        self.policy_id = "reference-compiler-v1"

    def __call__(self, context: PolicyContext) -> str:
        if self._index >= len(self._script):
            return json.dumps({"action": "final_answer",
                               "content": "Done (reference script exhausted)."})
        action = self._script[self._index]
        self._index += 1
        if action["action"] == "tool_call":
            outputs = _tool_outputs(context)
            action = {**action, "arguments": resolve_prev(action.get("arguments", {}), outputs)}
        return json.dumps(action, ensure_ascii=False)


def _tool_outputs(context: PolicyContext) -> list[Any]:
    """Parsed ``output`` payloads of every tool observation so far."""
    outputs = []
    for message in context.messages:
        if message["role"] != "tool":
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("ok") and "output" in payload:
            outputs.append(payload["output"])
    return outputs


# --------------------------------------------------------------------------
# Collection stats
# --------------------------------------------------------------------------

@dataclass
class CollectionStats:
    attempted: int = 0
    verified: int = 0            # rollouts that passed the verifier (regardless of caps)
    kept: int = 0                # trajectories actually retained
    dropped_by_cap: int = 0      # verified but discarded due to max_kept_per_task
    rejected: int = 0
    rejected_by_category: Counter = field(default_factory=Counter)
    failure_categories: Counter = field(default_factory=Counter)
    uncovered: list[tuple[str, str, str]] = field(default_factory=list)  # (task_id, category, reason)
    by_source: Counter = field(default_factory=Counter)

    def merge(self, other: "CollectionStats") -> None:
        self.attempted += other.attempted
        self.verified += other.verified
        self.kept += other.kept
        self.dropped_by_cap += other.dropped_by_cap
        self.rejected += other.rejected
        self.rejected_by_category.update(other.rejected_by_category)
        self.failure_categories.update(other.failure_categories)
        self.uncovered.extend(other.uncovered)
        self.by_source.update(other.by_source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted, "verified": self.verified, "kept": self.kept,
            "dropped_by_cap": self.dropped_by_cap, "rejected": self.rejected,
            "rejection_rate": (self.rejected / self.attempted) if self.attempted else None,
            "verified_rate": (self.verified / self.attempted) if self.attempted else None,
            "rejected_by_category": dict(self.rejected_by_category.most_common()),
            "failure_categories": dict(self.failure_categories.most_common()),
            "uncovered": [{"task_id": t, "category": c, "reason": r} for t, c, r in self.uncovered],
            "by_source": dict(self.by_source),
        }


def _guard_split(task: Task) -> None:
    if task.split == "test":
        raise DataGenerationError(
            f"Refusing to collect trajectories for test-split task {task.task_id}; "
            "the test split is reserved for final evaluation")


# --------------------------------------------------------------------------
# Reference trajectories
# --------------------------------------------------------------------------

def generate_reference_trajectories(tasks: Iterable[Task], executor: Executor, *,
                                    reward_config: RewardConfig | None = None,
                                    tools_fingerprint: str = "",
                                    run_tag: str = "ref") -> tuple[list[Trajectory], CollectionStats]:
    """Compile, execute, and verify reference solutions for training-split tasks."""
    trajectories: list[Trajectory] = []
    stats = CollectionStats()
    for index, task in enumerate(tasks):
        _guard_split(task)
        stats.attempted += 1
        script = compile_reference_solution(task)
        if script is None:
            reason = last_compilation_failure_reason()
            stats.uncovered.append((task.task_id, task.category, reason))
            logger.debug("no reference solution for %s (%s): %s", task.task_id, task.category, reason)
            continue
        episode = executor.run(task, SolutionPolicy(script))
        verdict = verify_episode(task, episode)
        if not verdict.success:
            stats.rejected += 1
            stats.rejected_by_category[task.category] += 1
            stats.failure_categories.update(verdict.failure_categories)
            logger.warning("compiled solution for %s failed verification: %s",
                           task.task_id, verdict.failure_categories)
            continue
        reward = compute_reward(episode, verdict, reward_config,
                                max_steps=task.max_steps).to_dict()
        trajectories.append(trajectory_from_episode(
            task, episode, verdict, source="reference",
            policy_id="reference-compiler-v1", sampling_seed=0, reward=reward,
            tools_fingerprint=tools_fingerprint, run_tag=run_tag, index=index))
        stats.verified += 1
        stats.kept += 1
        stats.by_source["reference"] += 1
    logger.info("Reference collection: %d/%d verified (%d uncovered, %d rejected)",
                stats.kept, stats.attempted, len(stats.uncovered), stats.rejected)
    return trajectories, stats


# --------------------------------------------------------------------------
# Model rollouts with rejection sampling
# --------------------------------------------------------------------------

PolicyFactory = Callable[[int], Any]


def collect_rollouts(tasks: Iterable[Task], executor: Executor, policy_factory: PolicyFactory, *,
                     samples_per_task: int = 4, source: str = "self",
                     reward_config: RewardConfig | None = None,
                     tools_fingerprint: str = "", run_tag: str = "",
                     base_seed: int = 0, max_kept_per_task: int | None = None,
                     keep_rejections: bool = False) -> tuple[list[Trajectory], CollectionStats]:
    """Rejection sampling: roll out a policy, keep verifier-passed episodes.

    Args:
        samples_per_task: rollouts per task (GRPO-style group size for later reuse).
        source: trajectory label ("self" | "teacher" | "random").
        max_kept_per_task: cap kept successes per task (None = keep all) —
            prevents one easy task from dominating the SFT mix.
        keep_rejections: also return rejected trajectories (labeled
            ``verified=False``) for analysis; SFT mixing filters them anyway.
    """
    if source not in ("self", "teacher", "random"):
        raise DataGenerationError(f"invalid rollout source {source!r}")
    trajectories: list[Trajectory] = []
    rejected: list[Trajectory] = []
    stats = CollectionStats()
    task_list = list(tasks)
    for task in task_list:
        _guard_split(task)
        kept_for_task = 0
        # Deterministic per-task seed offset (builtin hash() is salted per process).
        task_offset = int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16) % 10_000
        for sample in range(samples_per_task):
            seed = base_seed + 1000 * task_offset + sample
            policy = policy_factory(seed)
            set_seed = getattr(policy, "set_sampling", None)
            if callable(set_seed):
                set_seed(seed=seed)
            stats.attempted += 1
            episode: EpisodeResult = executor.run(task, policy)
            verdict = verify_episode(task, episode)
            reward = compute_reward(episode, verdict, reward_config,
                                    max_steps=task.max_steps).to_dict()
            traj = trajectory_from_episode(
                task, episode, verdict, source=source,
                policy_id=getattr(policy, "policy_id", type(policy).__name__),
                sampling_seed=seed, reward=reward, tools_fingerprint=tools_fingerprint,
                run_tag=run_tag or source, index=sample)
            if verdict.success:
                stats.verified += 1
                if max_kept_per_task is None or kept_for_task < max_kept_per_task:
                    trajectories.append(traj)
                    kept_for_task += 1
                    stats.kept += 1
                    stats.by_source[source] += 1
                else:
                    stats.dropped_by_cap += 1
            else:
                stats.rejected += 1
                stats.rejected_by_category[task.category] += 1
                stats.failure_categories.update(verdict.failure_categories)
                if keep_rejections:
                    rejected.append(traj)
    logger.info("Rollout collection (%s): attempted=%d verified=%d kept=%d rejected=%d "
                "dropped_by_cap=%d; failure categories: %s",
                source, stats.attempted, stats.verified, stats.kept, stats.rejected,
                stats.dropped_by_cap, dict(stats.failure_categories.most_common(5)))
    return (trajectories + rejected) if keep_rejections else trajectories, stats


# --------------------------------------------------------------------------
# SFT mixing
# --------------------------------------------------------------------------

def build_sft_mix(reference: Iterable[Trajectory], *more_sources: Iterable[Trajectory],
                  held_out_families: Iterable[str] = ("crm",),
                  max_per_task: int | None = None) -> tuple[list[Trajectory], dict[str, Any]]:
    """Merge trajectory sources into the SFT training pool with hygiene guards.

    - Verified successes only.
    - Test-split and held-out-family trajectories are refused (raised), since
      reference generation already excludes them — a violation here means an
      upstream bug, not something to filter silently.
    - ``max_per_task`` caps duplicates per task across sources (reference data
      is kept first, so caps never starve the deterministic seed set).
    """
    held_out = set(held_out_families)
    pool: list[Trajectory] = []
    per_task: Counter = Counter()
    dropped = Counter()
    ordered = list(reference) + [t for src in more_sources for t in src]
    for traj in ordered:
        if traj.split == "test" or traj.family in held_out:
            raise DataGenerationError(
                f"Trajectory {traj.trajectory_id} (split={traj.split}, family={traj.family}) "
                "must never enter SFT data")
        if not traj.verified:
            dropped["unverified"] += 1
            continue
        if traj.parse_error_count > 0 or traj.invalid_call_count > 0:
            dropped["not_clean"] += 1
            continue
        if max_per_task is not None and per_task[traj.task_id] >= max_per_task:
            dropped["per_task_cap"] += 1
            continue
        per_task[traj.task_id] += 1
        pool.append(traj)
    mix_stats = {
        "pool_size": len(pool),
        "unique_tasks": len(per_task),
        "by_source": dict(Counter(t.source for t in pool)),
        "by_family": dict(Counter(t.family for t in pool)),
        "by_category": dict(Counter(t.category for t in pool)),
        "dropped": dict(dropped),
    }
    logger.info("SFT mix: %d trajectories over %d tasks %s",
                mix_stats["pool_size"], mix_stats["unique_tasks"], mix_stats["by_source"])
    return pool, mix_stats


def write_collection(out_dir: str | Path, name: str, trajectories: list[Trajectory],
                     stats: CollectionStats) -> Path:
    """Persist trajectories + stats side by side; returns the JSONL path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_path = out / f"{name}.jsonl"
    save_trajectories(traj_path, trajectories)
    (out / f"{name}_stats.json").write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %d trajectories -> %s", len(trajectories), traj_path)
    return traj_path
