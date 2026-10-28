"""Evaluation runner: one interface for base, SFT, SFT+RL, and API policies.

Every policy faces identical tasks, tool definitions, step limits, initial
states, and verifier rules — the only variable is the policy itself. Guarantees:

- **Preflight before spending compute**: every task's fixture loads, every
  check spec resolves against the initial state, and the tool catalog
  fingerprint is recorded. Broken tasks fail loudly *before* any model runs.
- **Resumable**: predictions stream to JSONL (flushed per task); an
  interrupted run resumes by skipping completed ``(task_id, run_index)`` pairs.
- **Reproducible**: records carry sampling seeds, policy ids, grammar version,
  tools fingerprint, and the full config snapshot; metrics recompute from
  saved predictions alone.
- **Honest about policy failures**: exceptions inside a policy are recorded as
  failed episodes with ``termination="policy_error"`` and logged; if more than
  ``policy_error_abort_threshold`` of tasks error, the run aborts rather than
  reporting metrics dominated by infrastructure failures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from toolforge import __version__, get_logger
from toolforge.executor import EpisodeResult, Executor, PolicyContext
from toolforge.inference import GenerationStats, policy_id, tools_fingerprint
from toolforge.metrics import summarize, write_summary
from toolforge.schemas import GRAMMAR_VERSION, ToolForgeError
from toolforge.tasks import Task
from toolforge.verifier import VerificationResult, verify_episode

logger = get_logger("evaluation")

PolicyFactory = Callable[[int, int], Any]
"""``(run_index, sampling_seed) -> policy``; a fresh or re-seeded policy per run."""


class EvaluationError(ToolForgeError):
    """Raised for evaluation misconfiguration or unrecoverable run failures."""


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def _trace_counters(episode: EpisodeResult) -> dict[str, int]:
    """Per-trace counters not already aggregated on EpisodeResult."""
    parsed_tool_call_steps = 0     # parsed as tool_call (executed, unknown, or invalid args)
    arg_fail_steps = 0             # parsed tool_call with schema violations
    assistant_chars = 0
    for step in episode.trace:
        assistant_chars += len(step.raw_output)
        action = step.action
        if action and action.get("action") == "tool_call":
            parsed_tool_call_steps += 1
            if step.validation_issues:
                arg_fail_steps += 1
    return {"parsed_tool_call_steps": parsed_tool_call_steps,
            "arg_validation_failure_steps": arg_fail_steps,
            "assistant_output_chars": assistant_chars}


def build_task_record(task: Task, episode: EpisodeResult, verdict: VerificationResult, *,
                      run_index: int, sampling_seed: int, pid: str,
                      include_states: bool = True,
                      gen_stats: GenerationStats | None = None,
                      error_message: str | None = None) -> dict[str, Any]:
    """Assemble the persisted per-task prediction record."""
    counters = _trace_counters(episode)
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "run_index": run_index,
        "sampling_seed": sampling_seed,
        "policy_id": pid,
        "split": task.split,
        "family": task.family,
        "tool_families": task.tool_families,
        "category": task.category,
        "difficulty": task.difficulty,
        "request": task.request,
        "success": verdict.success,
        "failure_categories": verdict.failure_categories,
        "termination": episode.termination,
        "expected_terminal_action": task.expected.terminal_action if task.expected else None,
        "expected_clarify": bool(task.expected and task.expected.terminal_action == "clarify"),
        "predicted_clarify": episode.termination == "clarify",
        "final_output": episode.final_output,
        "final_output_chars": len(episode.final_output or ""),
        "tool_call_count": episode.tool_call_count,
        "invalid_call_count": episode.invalid_call_count,
        "parse_error_count": episode.parse_error_count,
        "turns": len(episode.trace),
        **counters,
        "forbidden_state_change": "forbidden_state_change" in verdict.failure_categories,
        "n_forbidden_changes": len(verdict.forbidden_changes),
        "forbidden_changes": verdict.forbidden_changes[:50],
        "verdict_checks": [c.to_dict() for c in verdict.checks],
        "policy_seconds": round(episode.policy_seconds, 6),
        "env_seconds": round(episode.env_seconds, 6),
        "prompt_tokens": gen_stats.prompt_tokens if gen_stats else None,
        "completion_tokens": gen_stats.completion_tokens if gen_stats else None,
        "ttft_seconds": gen_stats.ttft_seconds if gen_stats else None,
        "warnings": episode.warnings,
        "error_message": error_message,
        "trace": [s.to_dict() for s in episode.trace],
    }
    if include_states:
        record["initial_states"] = episode.initial_states
        record["final_states"] = episode.final_states
    return record


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def preflight_tasks(tasks: list[Task], executor: Executor) -> dict[str, Any]:
    """Validate every task is runnable and verifiable before spending compute.

    For each task: build its router (loads the fixture — catches corrupt
    states), then execute every check spec against a synthetic zero-action
    episode on the *initial* state. The pass/fail result is irrelevant; a
    raised ``VerifierError`` means the task could never be scored.
    """
    from toolforge.verifier import VerifierError, evaluate_check

    if not tasks:
        raise EvaluationError("Preflight received zero tasks")
    fingerprints: dict[str, str] = {}
    for task in tasks:
        if task.expected is None:
            raise EvaluationError(f"Task {task.task_id} has no expected outcome (no valid verifier)")
        try:
            router = executor.build_router(task)  # raises on unknown family / bad fixture
        except (ValueError, KeyError, ToolForgeError) as exc:
            raise EvaluationError(f"Task {task.task_id}: fixture/family error: {exc}") from exc
        key = ",".join(sorted(task.tool_families))
        fp = tools_fingerprint(router.tool_specs())
        if key in fingerprints and fingerprints[key] != fp:
            raise EvaluationError(
                f"Tool catalog fingerprint differs across tasks for families [{key}]: "
                f"{fingerprints[key]} vs {fp}")
        fingerprints[key] = fp
        initial = router.snapshots()
        dummy = EpisodeResult(task_id=task.task_id, termination="final_answer",
                              final_output="", initial_states=initial, final_states=initial,
                              volatile_keys={f: sorted(k) for f, k in router.volatile_keys().items()})
        for spec in task.expected.checks:
            try:
                evaluate_check(spec, dummy, task.family)  # malformed specs raise
            except VerifierError as exc:
                raise EvaluationError(
                    f"Task {task.task_id} cannot be verified (bad check spec): {exc}") from exc
    logger.info("Preflight OK: %d tasks, tool-catalog fingerprints %s",
                len(tasks), fingerprints)
    return {"n_tasks": len(tasks), "tools_fingerprints": fingerprints}


# --------------------------------------------------------------------------
# Run configuration metadata
# --------------------------------------------------------------------------

@dataclass
class RunMetadata:
    run_name: str
    policy_description: str
    task_file: str | None
    split: str
    n_tasks: int
    repeat: int
    base_seed: int
    grammar_version: str = GRAMMAR_VERSION
    toolforge_version: str = __version__
    tools_fingerprints: dict[str, str] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return out


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def filter_tasks(tasks: Iterable[Task], *, split: str | None = None,
                 families: Iterable[str] | None = None,
                 categories: Iterable[str] | None = None,
                 task_ids: Iterable[str] | None = None,
                 limit: int | None = None) -> list[Task]:
    """Deterministic task selection (sorted by task_id) for an evaluation run."""
    selected = list(tasks)
    if split and split != "all":
        selected = [t for t in selected if t.split == split]
    if families:
        wanted = set(families)
        selected = [t for t in selected if t.family in wanted]
    if categories:
        wanted_c = set(categories)
        selected = [t for t in selected if t.category in wanted_c]
    if task_ids:
        wanted_ids = set(task_ids)
        selected = [t for t in selected if t.task_id in wanted_ids]
    selected.sort(key=lambda t: t.task_id)
    if limit is not None:
        if limit < 1:
            raise EvaluationError(f"limit must be >= 1, got {limit}")
        selected = selected[:limit]
    if not selected:
        raise EvaluationError("Task selection matched zero tasks; check --split/--filters")
    return selected


def load_completed(predictions_path: Path) -> set[tuple[str, int]]:
    """(task_id, run_index) pairs already present in a predictions file."""
    done: set[tuple[str, int]] = set()
    if not predictions_path.exists():
        return done
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["task_id"], int(rec.get("run_index", 0))))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise EvaluationError(
                    f"Corrupt line in {predictions_path}; cannot resume safely: {exc}") from exc
    return done


def run_evaluation(tasks: list[Task], executor: Executor, policy_factory: PolicyFactory, *,
                   out_dir: str | Path, run_name: str, run_metadata: RunMetadata,
                   repeat: int = 1, base_seed: int = 0, include_states: bool = True,
                   resume: bool = True, progress_every: int = 10,
                   policy_error_abort_threshold: float = 0.5,
                   bootstrap_iterations: int = 10_000) -> dict[str, Any]:
    """Run all episodes, verify them, persist predictions, aggregate metrics.

    Returns the metrics summary dict (also written to ``metrics.json`` /
    ``metrics.csv`` next to ``predictions.jsonl`` and ``run_config.json``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = out / "predictions.jsonl"

    pre = preflight_tasks(tasks, executor)
    run_metadata.tools_fingerprints = pre["tools_fingerprints"]
    run_metadata.n_tasks = len(tasks)
    run_metadata.repeat = repeat
    run_metadata.base_seed = base_seed

    predictions_exists = predictions_path.exists()
    completed = load_completed(predictions_path) if resume else set()
    if predictions_exists and not resume:
        raise EvaluationError(
            f"{predictions_path} exists; pass resume=True or choose a new run dir "
            "(refusing to silently append to or overwrite prior predictions)")
    if completed:
        logger.info("Resuming: %d (task, run) pairs already complete", len(completed))

    fh = predictions_path.open("a", encoding="utf-8")
    records: list[dict[str, Any]] = []
    # Records restored from disk also count toward the summary (resume-safe).
    if completed:
        with predictions_path.open("r", encoding="utf-8") as rh:
            for line in rh:
                if line.strip():
                    records.append(json.loads(line))

    total = len(tasks) * repeat
    done_count = len(completed)
    policy_errors = 0
    started = time.time()
    try:
        for run_index in range(repeat):
            sampling_seed = base_seed + run_index
            pending = [t for t in tasks if (t.task_id, run_index) not in completed]
            if not pending:
                continue  # fully resumed run: never construct (or load) a policy
            policy = policy_factory(run_index, sampling_seed)
            pid = policy_id(policy)
            reset = getattr(policy, "reset_stats", None)
            for task in pending:
                if callable(reset):
                    reset()
                error_message = None
                try:
                    episode = executor.run(task, policy)
                except Exception as exc:  # policy/infrastructure failure, recorded loudly
                    logger.exception("Policy error on task %s (run %d)", task.task_id, run_index)
                    policy_errors += 1
                    error_message = f"{type(exc).__name__}: {exc}"
                    episode = EpisodeResult(task_id=task.task_id, termination="policy_error",
                                            warnings=[error_message])
                    # Verifier needs states; capture empty snapshots so the
                    # record is complete and verify_episode can run.
                    router = executor.build_router(task)
                    episode.initial_states = router.snapshots()
                    episode.final_states = router.snapshots()
                    episode.volatile_keys = {f: sorted(k) for f, k in router.volatile_keys().items()}
                stats = getattr(policy, "stats", None)
                verdict = verify_episode(task, episode)
                record = build_task_record(
                    task, episode, verdict, run_index=run_index, sampling_seed=sampling_seed,
                    pid=pid, include_states=include_states,
                    gen_stats=stats if isinstance(stats, GenerationStats) else None,
                    error_message=error_message)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(record)
                done_count += 1
                if policy_errors / max(done_count, 1) > policy_error_abort_threshold \
                        and policy_errors >= 5:
                    raise EvaluationError(
                        f"Aborting: {policy_errors}/{done_count} episodes failed with policy "
                        "errors (infrastructure problem, not model quality). Fix the policy "
                        "before trusting any metrics.")
                if progress_every and done_count % progress_every == 0:
                    elapsed = time.time() - started
                    ok = sum(1 for r in records if r["success"])
                    logger.info("progress %d/%d (%.0fs elapsed, running success %d/%d)",
                                done_count, total, elapsed, ok, len(records))
    finally:
        fh.close()

    elapsed = time.time() - started
    run_metadata.extra = {**run_metadata.extra,
                          "policy_errors": policy_errors,
                          "wall_seconds": round(elapsed, 2),
                          "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    summary = summarize(records, bootstrap_seed=base_seed,
                        bootstrap_iterations=bootstrap_iterations,
                        run_metadata=run_metadata.to_dict())
    write_summary(summary, str(out / "metrics.json"), str(out / "metrics.csv"))
    (out / "run_config.json").write_text(
        json.dumps(run_metadata.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Evaluation complete: %d records in %.1fs -> %s", len(records), elapsed, out)
    return summary


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load saved predictions for re-aggregation (metrics must be recomputable)."""
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"Predictions file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"{path}:{line_no}: corrupt JSON: {exc}") from exc
    if not records:
        raise EvaluationError(f"{path}: no records")
    return records


def reaggregate(predictions_path: str | Path, out_dir: str | Path, *,
                held_out_families: Iterable[str] = ("crm",),
                bootstrap_seed: int = 0, bootstrap_iterations: int = 10_000) -> dict[str, Any]:
    """Recompute metrics from saved predictions alone (reproducibility check)."""
    records = load_predictions(predictions_path)
    summary = summarize(records, held_out_families=held_out_families,
                        bootstrap_seed=bootstrap_seed, bootstrap_iterations=bootstrap_iterations)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(summary, str(out / "metrics.json"), str(out / "metrics.csv"))
    return summary


def make_context_stub(task: Task, executor: Executor) -> PolicyContext:
    """Convenience for warm-up code that needs a valid first-turn context."""
    _, context = executor.initial_context(task)
    return context
