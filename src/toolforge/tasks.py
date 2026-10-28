"""Benchmark task model, deterministic split assignment, and JSONL I/O.

A :class:`Task` is fully self-contained: request text, an explicit fixture
(the complete initial environment state), the tool families on offer, step
budget, difficulty/category metadata, split assignment, and an
:class:`ExpectedOutcome` — a declarative, JSON-serializable verification spec
consumed by :mod:`toolforge.verifier`. Expected outcomes are NEVER shown to
the model; the executor only sees request + tool schemas.

Split rules (deterministic, leak-free):

- Every task in a held-out family (default: ``crm``) goes to ``test``. The
  model may see those tools' schemas at evaluation but never trains on them.
- Remaining tasks are bucketed by a stable hash of ``(benchmark salt, task_id)``
  into train/val/test using the configured fractions. Hash-based assignment is
  order-independent and reproducible without storing RNG state.

Task content itself lives in :mod:`toolforge.tasks_builders`; this module is
the schema + plumbing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from toolforge.schemas import ToolForgeError

# --- task categories (the benchmark's 12 evaluation dimensions) --------------

CAT_SINGLE_STEP = "single_step_selection"
CAT_ARGUMENTS = "argument_construction"
CAT_MULTI_STEP = "multi_step_workflow"
CAT_RETRIEVE_ACT = "retrieve_then_act"
CAT_DEPENDENT = "dependent_actions"
CAT_AMBIGUOUS = "ambiguous_clarify"
CAT_NO_ACTION = "missing_info_no_action"
CAT_INVALID_TOOL = "invalid_tool_bait"
CAT_INDIRECT = "indirect_reference"
CAT_ERROR_RECOVERY = "error_recovery"
CAT_CONFLICT = "conflicting_constraints"
CAT_IMPOSSIBLE = "impossible_task"

CATEGORIES: tuple[str, ...] = (
    CAT_SINGLE_STEP, CAT_ARGUMENTS, CAT_MULTI_STEP, CAT_RETRIEVE_ACT,
    CAT_DEPENDENT, CAT_AMBIGUOUS, CAT_NO_ACTION, CAT_INVALID_TOOL,
    CAT_INDIRECT, CAT_ERROR_RECOVERY, CAT_CONFLICT, CAT_IMPOSSIBLE,
)

_CATEGORY_ABBR = {
    CAT_SINGLE_STEP: "ss", CAT_ARGUMENTS: "args", CAT_MULTI_STEP: "ms",
    CAT_RETRIEVE_ACT: "rt", CAT_DEPENDENT: "dep", CAT_AMBIGUOUS: "amb",
    CAT_NO_ACTION: "noact", CAT_INVALID_TOOL: "bait", CAT_INDIRECT: "ind",
    CAT_ERROR_RECOVERY: "rec", CAT_CONFLICT: "conf", CAT_IMPOSSIBLE: "imp",
}

DIFFICULTY_BY_CATEGORY = {
    CAT_SINGLE_STEP: "easy", CAT_ARGUMENTS: "easy", CAT_RETRIEVE_ACT: "easy",
    CAT_MULTI_STEP: "medium", CAT_DEPENDENT: "medium", CAT_INDIRECT: "medium",
    CAT_NO_ACTION: "medium", CAT_AMBIGUOUS: "medium",
    CAT_ERROR_RECOVERY: "hard", CAT_CONFLICT: "hard", CAT_INVALID_TOOL: "hard",
    CAT_IMPOSSIBLE: "hard",
}

DEFAULT_MAX_STEPS_BY_CATEGORY = {
    CAT_SINGLE_STEP: 3, CAT_ARGUMENTS: 3, CAT_RETRIEVE_ACT: 4,
    CAT_MULTI_STEP: 6, CAT_DEPENDENT: 6, CAT_INDIRECT: 5,
    CAT_AMBIGUOUS: 2, CAT_NO_ACTION: 3, CAT_INVALID_TOOL: 3,
    CAT_ERROR_RECOVERY: 6, CAT_CONFLICT: 5, CAT_IMPOSSIBLE: 3,
}

TERMINAL_ACTIONS = ("final_answer", "clarify", "any")

SPLITS = ("train", "val", "test")


class TaskError(ToolForgeError):
    """Raised for malformed tasks, fixtures, or split configurations."""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpectedOutcome:
    """Declarative verification spec for one task.

    Attributes:
        terminal_action: how a correct episode must end — ``"final_answer"``,
            ``"clarify"``, or ``"any"`` (either terminal action).
        forbid_state_changes: no non-volatile state mutation is allowed
            (read-only, no-action, clarify, and impossible tasks).
        checks: list of check specs evaluated by :mod:`toolforge.verifier`.
        max_invalid_calls: cap on parse/schema/unknown-tool failures
            (``None`` = uncapped; counts are always reported as metrics).
    """

    terminal_action: str = "final_answer"
    forbid_state_changes: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)
    max_invalid_calls: int | None = None

    def __post_init__(self) -> None:
        if self.terminal_action not in TERMINAL_ACTIONS:
            raise TaskError(f"terminal_action must be one of {TERMINAL_ACTIONS}, "
                            f"got {self.terminal_action!r}")
        for spec in self.checks:
            if not isinstance(spec, dict) or "check" not in spec:
                raise TaskError(f"Every check must be an object with a 'check' key, got {spec!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal_action": self.terminal_action,
            "forbid_state_changes": self.forbid_state_changes,
            "checks": self.checks,
            "max_invalid_calls": self.max_invalid_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedOutcome":
        return cls(
            terminal_action=data.get("terminal_action", "final_answer"),
            forbid_state_changes=bool(data.get("forbid_state_changes", False)),
            checks=list(data.get("checks", [])),
            max_invalid_calls=data.get("max_invalid_calls"),
        )


@dataclass
class Task:
    """One benchmark task; fully self-contained and JSON-serializable."""

    task_id: str
    family: str                      # primary environment family
    request: str                     # user-visible instruction
    category: str
    difficulty: str
    fixture: dict[str, Any]          # explicit initial state (per family if multi-family)
    fixture_seed: int                # seed the fixture was generated from (provenance)
    expected: ExpectedOutcome
    split: str = "unassigned"
    tool_families: list[str] = field(default_factory=list)  # defaults to [family]
    max_steps: int = 8
    tags: list[str] = field(default_factory=list)
    source: str = "template"         # "template" | "handwritten"
    benchmark_seed: int = 0
    benchmark_version: str = "v1"

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise TaskError(f"Task {self.task_id}: unknown category {self.category!r}")
        if not self.tool_families:
            self.tool_families = [self.family]
        if self.family not in self.tool_families:
            raise TaskError(f"Task {self.task_id}: family {self.family!r} not in tool_families")
        if self.max_steps < 1:
            raise TaskError(f"Task {self.task_id}: max_steps must be >= 1")
        if not isinstance(self.fixture, dict):
            raise TaskError(f"Task {self.task_id}: fixture must be an explicit state dict")
        if not self.request.strip():
            raise TaskError(f"Task {self.task_id}: request must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "family": self.family, "request": self.request,
            "category": self.category, "difficulty": self.difficulty,
            "fixture": self.fixture, "fixture_seed": self.fixture_seed,
            "expected": self.expected.to_dict(), "split": self.split,
            "tool_families": self.tool_families, "max_steps": self.max_steps,
            "tags": self.tags, "source": self.source,
            "benchmark_seed": self.benchmark_seed, "benchmark_version": self.benchmark_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"], family=data["family"], request=data["request"],
            category=data["category"], difficulty=data["difficulty"],
            fixture=data["fixture"], fixture_seed=data["fixture_seed"],
            expected=ExpectedOutcome.from_dict(data["expected"]),
            split=data.get("split", "unassigned"),
            tool_families=list(data.get("tool_families", [])),
            max_steps=int(data.get("max_steps", 8)),
            tags=list(data.get("tags", [])), source=data.get("source", "template"),
            benchmark_seed=int(data.get("benchmark_seed", 0)),
            benchmark_version=data.get("benchmark_version", "v1"),
        )


# --------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------

def _bucket(task_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{task_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_splits(tasks: list[Task], held_out_families: Iterable[str],
                  train_fraction: float, val_fraction: float,
                  salt: str) -> list[Task]:
    """Assign ``split`` on each task in place and return the list.

    Held-out families go entirely to ``test``. Others are bucketed by stable
    hash so splits never depend on iteration order or RNG state.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise TaskError("fractions must satisfy 0 < train < 1 and 0 <= val < 1")
    if train_fraction + val_fraction >= 1.0:
        raise TaskError("train_fraction + val_fraction must be < 1.0")
    held_out = set(held_out_families)
    for task in tasks:
        if task.family in held_out:
            task.split = "test"
            if "held_out_family" not in task.tags:
                task.tags.append("held_out_family")
            continue
        b = _bucket(task.task_id, salt)
        task.split = "train" if b < train_fraction else ("val" if b < train_fraction + val_fraction else "test")
    return tasks


def balance_category_coverage(tasks: list[Task], held_out_families: Iterable[str],
                              salt: str) -> list[Task]:
    """Guarantee every category has at least one test task (deterministically).

    Hash bucketing alone can leave a category absent from the test split, which
    would make per-category test metrics undefined. For each missing category we
    move exactly one train task (else val) to test: the one whose bucket value is
    largest, i.e. closest to the test boundary — a minimal, reproducible
    perturbation. Held-out family tasks are already all in test.
    """
    held_out = set(held_out_families)
    for cat in CATEGORIES:
        if any(t.split == "test" and t.category == cat for t in tasks):
            continue
        movable = [t for t in tasks
                   if t.category == cat and t.family not in held_out and t.split != "test"]
        if not movable:
            raise TaskError(
                f"Category {cat!r} has no test task and no movable (non-held-out) task; "
                "benchmark builders must produce enough tasks per category")
        from_train = [t for t in movable if t.split == "train"] or movable
        victim = max(from_train, key=lambda t: (_bucket(t.task_id, salt), t.task_id))
        victim.split = "test"
    return tasks


def check_no_leakage(tasks: list[Task], held_out_families: Iterable[str]) -> None:
    """Fail loudly on any split violation. Run at build time and in tests."""
    held_out = set(held_out_families)
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise TaskError(f"Duplicate task ids: {dupes[:10]}")
    by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    for task in tasks:
        if task.split not in SPLITS:
            raise TaskError(f"Task {task.task_id}: invalid split {task.split!r}")
        by_split[task.split].add(task.task_id)
        if task.family in held_out and task.split in ("train", "val"):
            raise TaskError(
                f"LEAKAGE: task {task.task_id} from held-out family {task.family!r} "
                f"assigned to {task.split!r}")
    overlap = ((by_split["train"] & by_split["val"])
               | (by_split["train"] & by_split["test"])
               | (by_split["val"] & by_split["test"]))
    if overlap:
        raise TaskError(f"LEAKAGE: task ids appear in multiple splits: {sorted(overlap)[:10]}")


# --------------------------------------------------------------------------
# I/O and summaries
# --------------------------------------------------------------------------

def save_tasks(path: str | Path, tasks: list[Task]) -> int:
    """Write tasks as JSONL (one task per line). Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(tasks)


def load_tasks(path: str | Path) -> list[Task]:
    """Load tasks from JSONL, validating each record."""
    path = Path(path)
    if not path.exists():
        raise TaskError(f"Task file not found: {path}")
    tasks = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            try:
                tasks.append(Task.from_dict(data))
            except (KeyError, TaskError) as exc:
                raise TaskError(f"{path}:{line_no}: invalid task record: {exc}") from exc
    if not tasks:
        raise TaskError(f"{path}: file contains no tasks")
    return tasks


def tasks_by_split(tasks: list[Task], split: str) -> list[Task]:
    return [t for t in tasks if t.split == split]


def benchmark_summary(tasks: list[Task]) -> dict[str, Any]:
    """Counts by split/category/family/difficulty for build reports."""
    def counts(key_fn) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in tasks:
            k = key_fn(t)
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    return {
        "total": len(tasks),
        "by_split": counts(lambda t: t.split),
        "by_category": counts(lambda t: t.category),
        "by_family": counts(lambda t: t.family),
        "by_difficulty": counts(lambda t: t.difficulty),
        "by_source": counts(lambda t: t.source),
        "category_x_split": counts(lambda t: f"{t.category}/{t.split}"),
        "family_x_split": counts(lambda t: f"{t.family}/{t.split}"),
    }
