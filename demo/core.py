"""Demo core: episode driving and verdict explanation — no Streamlit imports.

``demo/app.py`` is a thin UI over these functions; keeping the logic here means
the demo behavior is unit-testable headlessly (``tests/test_demo_core.py``).

Everything is simulated in-memory: no real emails, files, databases, network
calls, or secrets. Models are loaded ONLY when the UI explicitly asks (never at
import/startup time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolforge.config import EnvSettings
from toolforge.data_generation import SolutionPolicy
from toolforge.environments.registry import ENVIRONMENT_CLASSES
from toolforge.executor import Executor, ScriptedPolicy
from toolforge.reference_solutions import compile_reference_solution
from toolforge.registry import default_registry
from toolforge.tasks import ExpectedOutcome, Task, load_tasks
from toolforge.verifier import diff_states, verify_episode

DEFAULT_TASKS_FILE = "data/tasks/benchmark_v1.jsonl"

#: Plain-English explanations for every verifier failure category.
FAILURE_EXPLANATIONS = {
    "state_check_failed": "The final environment state does not match the required records/values "
                          "(checked against actual state, never text similarity).",
    "forbidden_state_change": "The task forbade state changes (read-only / clarify / refuse), "
                              "but the environment was modified.",
    "missing_clarification": "This request is genuinely ambiguous; the correct behavior is to "
                             "ask a clarifying question instead of acting.",
    "unnecessary_clarification": "The request was fully actionable; asking for clarification "
                                 "instead of acting fails the task.",
    "step_limit_exceeded": "The turn budget ran out before a terminal action "
                           "(final_answer/clarify) was produced.",
    "loop_guard": "The same tool call with identical arguments repeated too often; the "
                  "executor terminated the episode.",
    "required_tool_not_called": "A tool the task verifier expected to see was never called "
                                "(or not with the required arguments).",
    "forbidden_tool_called": "A tool the task forbids was called.",
    "answer_check_failed": "The final answer did not contain the computed ground-truth value(s).",
    "too_many_invalid_calls": "Too many malformed JSON / schema-violating / unknown-tool calls.",
    "no_termination": "The episode ended without a valid terminal action.",
    "policy_error": "The policy crashed while generating (infrastructure error, recorded loudly).",
}


@dataclass
class DemoStep:
    step: int
    raw_output: str
    action: dict[str, Any] | None
    parse_error: dict[str, str] | None
    tool: str | None
    arguments: dict[str, Any] | None
    result: dict[str, Any] | None


@dataclass
class DemoOutcome:
    request: str
    family: str
    task_id: str | None
    verified_task: bool          # True when a real benchmark task (with ground truth) was run
    termination: str
    final_output: str | None
    steps: list[DemoStep] = field(default_factory=list)
    success: bool = False
    failure_categories: list[str] = field(default_factory=list)
    check_lines: list[str] = field(default_factory=list)
    state_changes: list[str] = field(default_factory=list)
    reward: dict[str, Any] | None = None
    policy_seconds: float = 0.0
    env_seconds: float = 0.0


def available_families() -> list[str]:
    return sorted(ENVIRONMENT_CLASSES)


def list_benchmark_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE, *,
                         family: str | None = None, category: str | None = None,
                         limit: int = 25) -> list[dict[str, str]]:
    """Task menu entries: ids/requests/categories only — never expected outcomes."""
    tasks = load_tasks(tasks_file)
    if family:
        tasks = [t for t in tasks if t.family == family]
    if category:
        tasks = [t for t in tasks if t.category == category]
    return [{"task_id": t.task_id, "request": t.request, "family": t.family,
             "category": t.category, "difficulty": t.difficulty, "split": t.split}
            for t in sorted(tasks, key=lambda t: t.task_id)[:limit]]


def load_benchmark_task(task_id: str,
                        tasks_file: str | Path = DEFAULT_TASKS_FILE) -> Task:
    for task in load_tasks(tasks_file):
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id {task_id!r}")


def make_freeform_task(request: str, family: str, seed: int = 0) -> Task:
    """Ad-hoc request against a seeded default fixture (no ground truth).

    Freeform mode verifies only *clean completion* (a terminal action within the
    step budget); the demo says so explicitly instead of pretending to score
    correctness it cannot know.
    """
    env = ENVIRONMENT_CLASSES[family](seed=seed)
    return Task(task_id=f"freeform-{family}", family=family, request=request,
                category="single_step_selection", difficulty="easy",
                fixture=env.snapshot(), fixture_seed=seed,
                expected=ExpectedOutcome(terminal_action="any", checks=[]),
                max_steps=8, tags=["demo-freeform"], source="handwritten",
                benchmark_seed=seed, benchmark_version="demo")


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

def build_policy(mode: str, *, seed: int = 0, task: Task | None = None,
                 model_id: str | None = None, adapter_id: str | None = None,
                 compact_catalog: bool = False, model: Any = None,
                 tokenizer: Any = None) -> tuple[Any, str]:
    """(policy, human description). ``hf`` mode requires model+tokenizer to be
    passed in by the caller (the UI loads them lazily and caches them)."""
    if mode == "reference":
        if task is None:
            raise ValueError("reference policy needs a benchmark task")
        script = compile_reference_solution(task)
        if script is None:
            from toolforge.inference import RandomPolicy
            return RandomPolicy(seed=seed), "random (no compiled reference for this task)"
        return SolutionPolicy(script), "reference solution (compiled + verified)"
    if mode == "random":
        from toolforge.inference import RandomPolicy
        return RandomPolicy(seed=seed, max_tool_calls=4), f"random baseline (seed={seed})"
    if mode == "scripted-final":
        return ScriptedPolicy([{"action": "final_answer",
                                "content": "I cannot act without a real policy."}]), \
               "no-op policy (immediate final answer)"
    if mode == "hf":
        if model is None or tokenizer is None:
            raise ValueError("hf mode requires a pre-loaded model and tokenizer "
                             "(the demo never downloads at startup)")
        from toolforge.inference import HuggingFacePolicy
        desc = f"hf:{model_id}" + (f"+adapter:{Path(str(adapter_id)).name}" if adapter_id else "")
        return HuggingFacePolicy(model_id or "loaded-model", adapter_id=adapter_id,
                                 tokenizer=tokenizer, model=model, do_sample=False,
                                 max_new_tokens=384, enable_thinking=False,
                                 compact_catalog=compact_catalog), desc
    raise ValueError(f"Unknown demo policy mode {mode!r}")


# --------------------------------------------------------------------------
# Episode driving + explanation
# --------------------------------------------------------------------------

def run_demo(task: Task, policy: Any, *, compute_reward: bool = True,
             settings: EnvSettings | None = None) -> DemoOutcome:
    """Run one episode and package everything the UI displays."""
    executor = Executor(default_registry(), settings=settings or EnvSettings())
    episode = executor.run(task, policy)
    verdict = verify_episode(task, episode)
    reward = None
    if compute_reward:
        from toolforge.rewards import compute_reward as _reward
        reward = _reward(episode, verdict, max_steps=task.max_steps).to_dict()

    family_states_initial = episode.initial_states.get(task.family, {})
    family_states_final = episode.final_states.get(task.family, {})
    ignore = frozenset(episode.volatile_keys.get(task.family, ()))
    changes = diff_states(family_states_initial, family_states_final,
                          path=task.family, ignore_keys=ignore)

    steps = [DemoStep(step=s.step, raw_output=s.raw_output, action=s.action,
                      parse_error=s.parse_failure, tool=s.tool, arguments=s.arguments,
                      result=s.tool_result) for s in episode.trace]
    check_lines = []
    for c in verdict.checks:
        mark = "PASS" if c.passed else "FAIL"
        check_lines.append(f"[{mark}] {c.check}: {c.detail}")
    return DemoOutcome(
        request=task.request, family=task.family,
        task_id=None if task.task_id.startswith("freeform-") else task.task_id,
        verified_task=not task.task_id.startswith("freeform-"),
        termination=episode.termination, final_output=episode.final_output,
        steps=steps, success=verdict.success,
        failure_categories=list(verdict.failure_categories), check_lines=check_lines,
        state_changes=changes, reward=reward,
        policy_seconds=episode.policy_seconds, env_seconds=episode.env_seconds)


def explain_outcome(outcome: DemoOutcome) -> list[str]:
    """Human-readable verdict explanation (the demo's 'why it passed/failed')."""
    lines: list[str] = []
    if not outcome.verified_task:
        lines.append("Freeform request: no ground truth exists, so the demo only checks "
                     "clean completion (terminal action within the step budget). "
                     "Run a benchmark task for real verification.")
    status = "SUCCESS — verified against the actual final environment state." \
        if outcome.success else "FAILED verification."
    lines.append(status)
    lines.append(f"Termination: {outcome.termination}.")
    for category in outcome.failure_categories:
        explanation = FAILURE_EXPLANATIONS.get(category, "See verifier documentation.")
        lines.append(f"- {category}: {explanation}")
    if outcome.state_changes:
        lines.append(f"Environment changes ({len(outcome.state_changes)}):")
        lines.extend(f"  - {c}" for c in outcome.state_changes[:12])
    else:
        lines.append("Environment changes: none.")
    return lines


def render_policy_context_note() -> str:
    """One-paragraph grammar reference shown in the UI."""
    return ("The model emits exactly one JSON object per turn: "
            '{"action": "tool_call", "tool": "family.tool", "arguments": {...}}, '
            '{"action": "clarify", "question": "..."}, or '
            '{"action": "final_answer", "content": "..."}. Tool results return as '
            '{"ok": true, "output": ...} / {"ok": false, "error": {code, message}}. '
            "All state is in-memory and simulated.")


def outcome_to_dict(outcome: DemoOutcome) -> dict[str, Any]:
    """JSON-serializable view (used by tests and the UI's download button)."""
    return {
        "task_id": outcome.task_id, "request": outcome.request, "family": outcome.family,
        "verified_task": outcome.verified_task, "termination": outcome.termination,
        "final_output": outcome.final_output, "success": outcome.success,
        "failure_categories": outcome.failure_categories, "check_lines": outcome.check_lines,
        "state_changes": outcome.state_changes, "reward": outcome.reward,
        "steps": [{"step": s.step, "action": s.action, "tool": s.tool,
                   "arguments": s.arguments, "result": s.result,
                   "parse_error": s.parse_error} for s in outcome.steps],
        "policy_seconds": round(outcome.policy_seconds, 4),
        "env_seconds": round(outcome.env_seconds, 4),
    }
