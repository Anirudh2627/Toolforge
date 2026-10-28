"""Episode executor: validates, routes, and runs tool calls with full traces.

The executor is policy-agnostic. It drives any callable that maps a
:class:`PolicyContext` to raw text (a scripted list in tests, a model wrapper
in Phase 2+), so the exact same loop powers evaluation, trajectory collection,
GRPO rollouts, and the demo.

Termination reasons:

- ``final_answer`` / ``clarify``: the policy emitted that action.
- ``step_limit``: ``task.max_steps`` assistant turns consumed. Every turn
  counts, including parse failures and schema violations, so a model cannot
  stall by emitting garbage.
- ``loop_guard``: one fully identical ``(tool, arguments)`` call repeated more
  than ``repeat_limit`` times.

Error feedback (parse failures, unknown tools, schema violations, tool
errors) is returned to the policy as a ``tool``-role message containing the
same ``{"ok": false, "error": {...}}`` JSON shape as real tool results — one
consistent observation channel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from toolforge import get_logger
from toolforge.config import EnvSettings
from toolforge.registry import EnvironmentRegistry, ToolRouter
from toolforge.schemas import (
    Action,
    ActionKind,
    ParseFailure,
    parse_action,
    render_tool_catalog,
    validate_against_schema,
)

logger = get_logger("executor")

SYSTEM_PROMPT_TEMPLATE = """\
You are ToolForge, an assistant that completes user requests by calling tools.

Respond with EXACTLY ONE JSON object per turn and no other text. Choose one action:

1. Call a tool:
{{"action": "tool_call", "tool": "<tool_name>", "arguments": {{...}}}}

2. Ask the user for clarification, but only when the request is genuinely ambiguous or required information is missing and cannot be found with the tools:
{{"action": "clarify", "question": "<your question>"}}

3. Give the final answer when the request is complete or cannot be completed with the available tools:
{{"action": "final_answer", "content": "<your answer>"}}

Available tools (name, description, JSON-schema parameters):
{tool_catalog}

Tool results arrive as JSON: {{"ok": true, "output": ...}} or {{"ok": false, "error": {{"code": ..., "message": ...}}}}. \
If a call fails, read the error, adapt, and try a different approach. \
Never invent ids or values you have not observed in a tool result. \
The simulated current date is {current_date}. All operations are sandboxed; nothing affects the real world."""


@dataclass
class PolicyContext:
    """Everything the policy may see. Expected outcomes are never included."""

    request: str
    tools: list[dict[str, Any]]        # ToolSpec dicts (model-visible)
    messages: list[dict[str, str]]     # conversation so far (system/user/assistant/tool)
    step: int                          # 0-based index of the upcoming turn
    remaining_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {"request": self.request, "tools": self.tools, "messages": list(self.messages),
                "step": self.step, "remaining_steps": self.remaining_steps}


class Policy(Protocol):
    def __call__(self, context: PolicyContext) -> str: ...


class ScriptedPolicy:
    """Replays a fixed list of actions (dicts are serialized; strings pass through).

    Used by tests and reference trajectories. When the script runs out, emits
    a final_answer so episodes always terminate; the executor records this via
    the ``script_exhausted`` warning on the episode.
    """

    def __init__(self, actions: list[dict[str, Any] | str]) -> None:
        self._queue = list(actions)
        self.exhausted = False

    def __call__(self, context: PolicyContext) -> str:
        if not self._queue:
            self.exhausted = True
            return json.dumps({"action": "final_answer",
                               "content": "(script exhausted without a terminal action)"})
        item = self._queue.pop(0)
        return item if isinstance(item, str) else json.dumps(item)


@dataclass
class TraceStep:
    """One assistant turn and its consequence."""

    step: int
    raw_output: str
    action: dict[str, Any] | None = None          # parsed action dict, or None
    parse_failure: dict[str, str] | None = None   # ParseFailure.to_dict(), or None
    validation_issues: list[dict[str, str]] | None = None  # schema violations
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None     # ToolResult.to_dict()
    repeat_count: int = 0                          # times this (tool,args) was seen before
    policy_seconds: float = 0.0
    env_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "raw_output": self.raw_output, "action": self.action,
            "parse_failure": self.parse_failure, "validation_issues": self.validation_issues,
            "tool": self.tool, "arguments": self.arguments, "tool_result": self.tool_result,
            "repeat_count": self.repeat_count,
            "policy_seconds": round(self.policy_seconds, 6),
            "env_seconds": round(self.env_seconds, 6),
        }


@dataclass
class EpisodeResult:
    """Complete record of one episode: everything metrics and rewards need."""

    task_id: str
    termination: str                       # see module docstring
    final_output: str | None = None        # final_answer content or clarify question
    trace: list[TraceStep] = field(default_factory=list)
    initial_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    volatile_keys: dict[str, list[str]] = field(default_factory=dict)
    tool_call_count: int = 0               # executed calls (schema-valid, known tool)
    invalid_call_count: int = 0            # parse failures + schema violations + unknown tools
    parse_error_count: int = 0
    policy_seconds: float = 0.0
    env_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "termination": self.termination,
            "final_output": self.final_output,
            "trace": [t.to_dict() for t in self.trace],
            "initial_states": self.initial_states, "final_states": self.final_states,
            "volatile_keys": self.volatile_keys,
            "tool_call_count": self.tool_call_count,
            "invalid_call_count": self.invalid_call_count,
            "parse_error_count": self.parse_error_count,
            "policy_seconds": round(self.policy_seconds, 6),
            "env_seconds": round(self.env_seconds, 6),
            "warnings": self.warnings,
        }


class TaskHandle(Protocol):
    """The subset of :class:`toolforge.tasks.Task` the executor relies on.

    Declared as a protocol so the executor does not import the benchmark
    module (keeping the dependency direction tasks -> executor).
    """

    task_id: str
    request: str
    tool_families: list[str]
    fixture: dict[str, Any] | None
    fixture_seed: int
    max_steps: int


class Executor:
    """Runs episodes against isolated environment instances."""

    def __init__(self, registry: EnvironmentRegistry,
                 settings: EnvSettings | None = None,
                 env_options: dict[str, dict[str, Any]] | None = None) -> None:
        self._registry = registry
        self._settings = settings or EnvSettings()
        # Per-family constructor kwargs (e.g. {"sql": {"max_rows": 50}}).
        self._env_options = env_options or {}

    def build_router(self, task: TaskHandle) -> ToolRouter:
        """Create isolated environments for one episode from the task fixture."""
        environments = {}
        multi = len(task.tool_families) > 1
        for family in task.tool_families:
            fixture = None
            if task.fixture is not None:
                if multi:
                    if not isinstance(task.fixture, dict) or family not in task.fixture:
                        raise ValueError(
                            f"Task {task.task_id}: multi-family fixture must map every family "
                            f"to a state dict; missing {family!r}")
                    fixture = task.fixture[family]
                else:
                    fixture = task.fixture
            options = self._env_options.get(family, {})
            environments[family] = self._registry.create(
                family, fixture=fixture, seed=task.fixture_seed, **options)
        return ToolRouter(environments)

    def build_system_prompt(self, router: ToolRouter) -> str:
        specs = router.tool_specs()
        current_date = self._current_date(router)
        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_catalog=render_tool_catalog(specs), current_date=current_date)

    @staticmethod
    def _current_date(router: ToolRouter) -> str:
        for env in router.environments.values():
            date = env.state.get("current_date")
            if isinstance(date, str):
                return date
        return "unknown"

    def run(self, task: TaskHandle, policy: Policy) -> EpisodeResult:
        """Execute one full episode and return its trace and final states."""
        router = self.build_router(task)
        result = EpisodeResult(
            task_id=task.task_id,
            termination="step_limit",
            initial_states=router.snapshots(),
            volatile_keys={f: sorted(k) for f, k in router.volatile_keys().items()},
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(router)},
            {"role": "user", "content": task.request},
        ]
        tool_specs = [spec.to_dict() for spec in router.tool_specs()]
        seen_calls: dict[str, int] = {}
        max_steps = max(1, int(task.max_steps))

        for step in range(max_steps):
            context = PolicyContext(request=task.request, tools=tool_specs,
                                    messages=list(messages), step=step,
                                    remaining_steps=max_steps - step)
            t0 = time.perf_counter()
            raw = policy(context)
            policy_seconds = time.perf_counter() - t0
            if not isinstance(raw, str):
                raise TypeError(f"Policy must return str, got {type(raw).__name__}")

            trace_step = TraceStep(step=step, raw_output=raw, policy_seconds=policy_seconds)
            result.policy_seconds += policy_seconds
            parsed = parse_action(raw)

            if isinstance(parsed, ParseFailure):
                trace_step.parse_failure = parsed.to_dict()
                result.invalid_call_count += 1
                result.parse_error_count += 1
                messages.append({"role": "assistant", "content": raw})
                self._append_error(messages, trace_step, parsed.category, parsed.message)
                result.trace.append(trace_step)
                continue

            action: Action = parsed
            trace_step.action = action.to_dict()
            messages.append({"role": "assistant", "content": raw})

            if action.kind is ActionKind.FINAL_ANSWER:
                result.termination = "final_answer"
                result.final_output = action.content
                result.trace.append(trace_step)
                break
            if action.kind is ActionKind.CLARIFY:
                result.termination = "clarify"
                result.final_output = action.question
                result.trace.append(trace_step)
                break

            # --- tool_call ---
            assert action.tool is not None
            spec = router.spec_for(action.tool)
            if spec is None:
                trace_step.tool = action.tool
                trace_step.arguments = action.arguments
                result.invalid_call_count += 1
                known = router.known_tool_names()
                self._append_error(messages, trace_step, "unknown_tool",
                                   f"Tool {action.tool!r} does not exist. Available tools: {known}")
                result.trace.append(trace_step)
                continue

            issues = validate_against_schema(action.arguments or {}, spec.parameters, "arguments")
            if issues:
                trace_step.tool = action.tool
                trace_step.arguments = action.arguments
                trace_step.validation_issues = [i.to_dict() for i in issues]
                result.invalid_call_count += 1
                self._append_error(messages, trace_step, "schema_violation",
                                   "; ".join(f"{i.path}: {i.message}" for i in issues))
                result.trace.append(trace_step)
                continue

            # Loop guard on exact (tool, arguments) identity.
            call_key = json.dumps([action.tool, action.arguments], sort_keys=True, default=str)
            seen_calls[call_key] = seen_calls.get(call_key, 0) + 1
            trace_step.repeat_count = seen_calls[call_key] - 1

            env, local_name = router.resolve(action.tool)
            t1 = time.perf_counter()
            tool_result = env.execute(local_name, action.arguments or {})
            trace_step.env_seconds = time.perf_counter() - t1
            result.env_seconds += trace_step.env_seconds
            trace_step.tool = action.tool
            trace_step.arguments = action.arguments
            trace_step.tool_result = tool_result.to_dict()
            result.tool_call_count += 1
            messages.append({"role": "tool",
                             "content": json.dumps(tool_result.to_dict(), ensure_ascii=False)})
            result.trace.append(trace_step)

            if seen_calls[call_key] > self._settings.repeat_limit:
                result.termination = "loop_guard"
                result.warnings.append(
                    f"loop_guard: identical call {action.tool} repeated "
                    f"{seen_calls[call_key]} times (> {self._settings.repeat_limit})")
                break
        else:
            result.termination = "step_limit"
            result.warnings.append(f"step_limit: {max_steps} assistant turns consumed")

        result.final_states = router.snapshots()
        if isinstance(policy, ScriptedPolicy) and policy.exhausted:
            result.warnings.append("script_exhausted: ScriptedPolicy ran out of actions")
        return result

    @staticmethod
    def _append_error(messages: list[dict[str, str]], trace_step: TraceStep,
                      code: str, message: str) -> None:
        error_result = {"ok": False, "error": {"code": code, "message": message}}
        trace_step.tool_result = error_result
        messages.append({"role": "tool", "content": json.dumps(error_result, ensure_ascii=False)})


def execute_scripted(registry: EnvironmentRegistry, task: TaskHandle,
                     actions: list[dict[str, Any] | str],
                     settings: EnvSettings | None = None) -> EpisodeResult:
    """Convenience wrapper: run one episode with a :class:`ScriptedPolicy`."""
    return Executor(registry, settings=settings).run(task, ScriptedPolicy(actions))
