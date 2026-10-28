"""Execution verifier: task success is computed from environment state.

The verifier never compares a model's prose against a reference answer. It
inspects (a) the final environment state vs. the initial snapshot, (b) the
tool-call trace, and (c) — only where state cannot express correctness, e.g.
information-retrieval tasks — lexical containment of computed ground-truth
values in the final answer. All checks are declarative dicts stored on the
task (see :mod:`toolforge.tasks`), so every verdict is reproducible from the
saved episode record.

Check types (JSON-serializable, evaluated here):

- ``record_exists`` / ``record_absent``: records matching ``where`` must
  exist / not exist in a collection of the final state.
- ``record_count``: number of matching records ``op value`` (==, !=, >=, <=, >, <).
- ``field_equals``: exactly one record matches ``where``; its ``field``
  matches ``value`` (scalar or op-dict).
- ``no_overlaps``: no two records (after ``filter``) overlap on
  ``date_field``+``[start_field, end_field)`` — used for calendar correctness.
- ``tool_called`` / ``tool_not_called``: trace-based; ``arguments_match``
  optionally constrains the arguments of at least one matching call.
- ``answer_contains_all`` / ``answer_contains_any``: lexical checks against
  the final answer (only meaningful with ``final_answer`` termination).
- ``invalid_calls_at_most``: caps parse/schema/unknown-tool failures.

``where``/``value`` matching semantics (:func:`match_value`):
- plain value vs list field -> membership;
- list vs list -> order-insensitive equality;
- ``{"op": "contains"|"eq"|"ne"|"gt"|"gte"|"lt"|"lte"|"in"|"not_in"|"exists", "value": ...}``
  for explicit operators (``contains`` = substring for strings, membership for lists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolforge import get_logger
from toolforge.executor import EpisodeResult
from toolforge.timeutils import time_to_minutes

logger = get_logger("verifier")

_COUNT_OPS = {"==", "!=", ">=", "<=", ">", "<"}


class VerifierError(Exception):
    """Raised for malformed check specs or episodes that cannot be verified."""


# --------------------------------------------------------------------------
# Value / record matching
# --------------------------------------------------------------------------

def match_value(actual: Any, expected: Any) -> bool:
    """Compare an actual value against an expected literal or op-dict."""
    if isinstance(expected, dict) and "op" in expected:
        return _apply_op(actual, expected)
    if isinstance(actual, list) and isinstance(expected, list):
        try:
            return set(actual) == set(expected)
        except TypeError:  # unhashable items: fall back to order-insensitive compare
            return sorted(map(repr, actual)) == sorted(map(repr, expected))
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def _apply_op(actual: Any, spec: dict[str, Any]) -> bool:
    op = spec["op"]
    if op == "exists":
        return (actual is not None) is bool(spec.get("value", True))
    value = spec.get("value")
    if op == "contains":
        case_insensitive = spec.get("case_insensitive", True)
        if isinstance(actual, str) and isinstance(value, str):
            hay, needle = (actual.lower(), value.lower()) if case_insensitive else (actual, value)
            return needle in hay
        if isinstance(actual, list):
            return any(match_value(item, value) for item in actual)
        return False
    if op == "eq":
        return match_value(actual, value)
    if op == "ne":
        return not match_value(actual, value)
    if op in ("gt", "gte", "lt", "lte"):
        if actual is None or value is None:
            return False
        try:
            if op == "gt":
                return actual > value
            if op == "gte":
                return actual >= value
            if op == "lt":
                return actual < value
            return actual <= value
        except TypeError:
            return False
    if op == "in":
        return isinstance(value, list) and any(match_value(actual, v) for v in value)
    if op == "not_in":
        return isinstance(value, list) and not any(match_value(actual, v) for v in value)
    raise VerifierError(f"Unknown match op {op!r}")


def match_where(record: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(match_value(record.get(key), expected) for key, expected in where.items())


# --------------------------------------------------------------------------
# State diffing (forbidden side effects)
# --------------------------------------------------------------------------

def diff_states(initial: Any, final: Any, path: str = "",
                ignore_keys: frozenset[str] = frozenset()) -> list[str]:
    """Human-readable list of differences between two state snapshots.

    Lists of dicts with an ``id`` field are aligned by id (records are stable
    across episodes); other lists align by index. Keys named in
    ``ignore_keys`` (the environment's volatile keys) are skipped anywhere.
    """
    changes: list[str] = []
    if isinstance(initial, dict) and isinstance(final, dict):
        for key in sorted(set(initial) | set(final)):
            if key in ignore_keys:
                continue
            child = f"{path}.{key}" if path else key
            if key not in final:
                changes.append(f"{child}: removed (was {_brief(initial[key])})")
            elif key not in initial:
                changes.append(f"{child}: added ({_brief(final[key])})")
            else:
                changes.extend(diff_states(initial[key], final[key], child, ignore_keys))
    elif isinstance(initial, list) and isinstance(final, list):
        changes.extend(_diff_lists(initial, final, path, ignore_keys))
    elif initial != final:
        changes.append(f"{path or '<root>'}: {_brief(initial)} -> {_brief(final)}")
    return changes


def _diff_lists(initial: list, final: list, path: str,
                ignore_keys: frozenset[str]) -> list[str]:
    changes: list[str] = []
    if initial and final and all(isinstance(x, dict) and "id" in x for x in initial + final):
        by_id_initial = {x["id"]: x for x in initial}
        by_id_final = {x["id"]: x for x in final}
        for rid in sorted(set(by_id_initial) | set(by_id_final)):
            child = f"{path}[id={rid}]"
            if rid not in by_id_final:
                changes.append(f"{child}: record removed")
            elif rid not in by_id_initial:
                changes.append(f"{child}: record added")
            else:
                changes.extend(diff_states(by_id_initial[rid], by_id_final[rid], child, ignore_keys))
    else:
        if len(initial) != len(final):
            changes.append(f"{path}.length: {len(initial)} -> {len(final)}")
        for i, (a, b) in enumerate(zip(initial, final)):
            changes.extend(diff_states(a, b, f"{path}[{i}]", ignore_keys))
    return changes


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


# --------------------------------------------------------------------------
# Check evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    check: str
    params: dict[str, Any]
    passed: bool
    detail: str
    category: str  # failure category contributed when not passed ("" if passed)

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.check, "params": self.params, "passed": self.passed,
                "detail": self.detail, "category": self.category}


def _resolve_collection(states: dict[str, dict[str, Any]], primary_family: str,
                        collection: str) -> list[dict[str, Any]]:
    """Resolve "events" (primary family) or "crm.contacts" to a record list."""
    if "." in collection:
        family, key = collection.split(".", 1)
    else:
        family, key = primary_family, collection
    if family not in states:
        raise VerifierError(f"Collection {collection!r}: family {family!r} not in episode states "
                            f"({sorted(states)})")
    value = states[family].get(key)
    if not isinstance(value, list):
        raise VerifierError(f"Collection {collection!r} is not a list in family {family!r}")
    return value


def _executed_calls_including_errors(episode: EpisodeResult) -> list[dict[str, Any]]:
    """Executed calls regardless of tool-level ok/error (the call did happen)."""
    calls = []
    for step in episode.trace:
        if step.tool and step.tool_result and step.validation_issues is None \
                and step.parse_failure is None:
            calls.append({"tool": step.tool, "arguments": step.arguments or {},
                          "ok": step.tool_result.get("ok", False)})
    return calls


def evaluate_check(spec: dict[str, Any], episode: EpisodeResult,
                   primary_family: str) -> CheckResult:
    """Evaluate one declarative check against a finished episode."""
    if not isinstance(spec, dict) or "check" not in spec:
        raise VerifierError(f"Check spec must be an object with a 'check' key, got {spec!r}")
    check = spec["check"]
    params = {k: v for k, v in spec.items() if k != "check"}
    states = episode.final_states

    def result(passed: bool, detail: str, category: str) -> CheckResult:
        return CheckResult(check, params, passed, detail, "" if passed else category)

    if check in ("record_exists", "record_absent"):
        records = _resolve_collection(states, primary_family, spec["collection"])
        where = spec.get("where", {})
        matches = [r for r in records if match_where(r, where)]
        if check == "record_exists":
            return result(bool(matches),
                          f"{len(matches)} record(s) match {where}",
                          "state_check_failed")
        return result(not matches,
                      f"expected no match for {where}, found {len(matches)}: {_brief(matches[:2])}",
                      "state_check_failed")

    if check == "record_count":
        records = _resolve_collection(states, primary_family, spec["collection"])
        where = spec.get("where", {})
        n = sum(1 for r in records if match_where(r, where))
        op, expected = spec.get("op", "=="), spec["value"]
        if op not in _COUNT_OPS:
            raise VerifierError(f"record_count: unknown op {op!r}")
        passed = {"==": n == expected, "!=": n != expected, ">=": n >= expected,
                  "<=": n <= expected, ">": n > expected, "<": n < expected}[op]
        return result(passed, f"count={n}, expected {op} {expected}", "state_check_failed")

    if check == "field_equals":
        records = _resolve_collection(states, primary_family, spec["collection"])
        where = spec.get("where", {})
        matches = [r for r in records if match_where(r, where)]
        if len(matches) != 1:
            return result(False,
                          f"expected exactly 1 record matching {where}, got {len(matches)}",
                          "state_check_failed")
        actual = matches[0].get(spec["field"], "<missing>")
        passed = match_value(actual, spec["value"])
        return result(passed, f"field {spec['field']!r} = {_brief(actual)}, expected {spec['value']!r}",
                      "state_check_failed")

    if check == "no_overlaps":
        # Every record matching subject_where must not overlap any *other*
        # record matching filter. Scoped this way because fixtures may contain
        # pre-existing overlaps that are not the model's doing.
        records = _resolve_collection(states, primary_family, spec["collection"])
        subject_where = spec.get("subject_where")
        if not subject_where:
            raise VerifierError("no_overlaps requires 'subject_where'")
        subjects = [r for r in records if match_where(r, subject_where)]
        if not subjects:
            return result(False, f"no record matches subject_where {subject_where}",
                          "state_check_failed")
        flt = spec.get("filter")
        others = [r for r in records if (not flt or match_where(r, flt))]
        date_f = spec.get("date_field", "date")
        start_f = spec.get("start_field", "start")
        end_f = spec.get("end_field", "end")

        def _minutes(record: dict, key: str) -> int:
            try:
                return time_to_minutes(record[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise VerifierError(
                    f"no_overlaps: record {record.get('id', '?')} has unusable {key!r}: {exc}") from exc

        overlaps = []
        for subj in subjects:
            s_date, s_start, s_end = subj.get(date_f), _minutes(subj, start_f), _minutes(subj, end_f)
            for other in others:
                if other is subj or other.get(date_f) != s_date:
                    continue
                if s_start < _minutes(other, end_f) and _minutes(other, start_f) < s_end:
                    overlaps.append(f"{s_date}: {subj.get('id')} [{subj.get(start_f)}-{subj.get(end_f)}] "
                                    f"vs {other.get('id')} [{other.get(start_f)}-{other.get(end_f)}]")
        return result(not overlaps,
                      "subject records overlap nothing" if not overlaps else f"overlaps: {overlaps[:4]}",
                      "state_check_failed")

    if check == "tool_called":
        tool = spec["tool"]
        calls = [c for c in _executed_calls_including_errors(episode) if c["tool"] == tool]
        min_times = spec.get("min_times", 1)
        max_times = spec.get("max_times")
        if len(calls) < min_times or (max_times is not None and len(calls) > max_times):
            return result(False, f"{tool} called {len(calls)} time(s), expected "
                                 f">={min_times}" + (f", <={max_times}" if max_times is not None else ""),
                          "required_tool_not_called")
        args_match = spec.get("arguments_match")
        if args_match is not None:
            if not any(match_where(c["arguments"], args_match) for c in calls):
                return result(False,
                              f"{tool} was called but never with arguments matching {args_match}",
                              "required_tool_not_called")
        return result(True, f"{tool} called {len(calls)} time(s)", "")

    if check == "tool_not_called":
        tool = spec["tool"]
        calls = [c for c in _executed_calls_including_errors(episode) if c["tool"] == tool]
        return result(not calls, f"{tool} was called {len(calls)} time(s)", "forbidden_tool_called")

    if check in ("answer_contains_all", "answer_contains_any"):
        needles = spec["strings"]
        if not isinstance(needles, list) or not needles or not all(isinstance(s, str) for s in needles):
            raise VerifierError(f"{check}: 'strings' must be a non-empty list of strings")
        answer = episode.final_output or ""
        if spec.get("case_insensitive", True):
            answer = answer.lower()
            needles = [s.lower() for s in needles]
        if check == "answer_contains_all":
            missing = [s for s in needles if s not in answer]
            return result(not missing,
                          "all required strings present" if not missing else f"missing: {missing}",
                          "answer_check_failed")
        present = [s for s in needles if s in answer]
        return result(bool(present),
                      f"found {present[:3]}" if present else f"none of {needles} present",
                      "answer_check_failed")

    if check == "invalid_calls_at_most":
        limit = spec["n"]
        return result(episode.invalid_call_count <= limit,
                      f"invalid_call_count={episode.invalid_call_count}, limit={limit}",
                      "too_many_invalid_calls")

    raise VerifierError(f"Unknown check type {check!r} in spec {spec}")


# --------------------------------------------------------------------------
# Episode verification
# --------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Full verdict for one episode; JSON-serializable for reproducible reports."""

    task_id: str
    success: bool
    failure_categories: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    forbidden_changes: list[str] = field(default_factory=list)
    termination: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "failure_categories": self.failure_categories,
            "checks": [c.to_dict() for c in self.checks],
            "forbidden_changes": self.forbidden_changes,
            "termination": self.termination,
            "stats": self.stats,
        }


_BAD_TERMINATIONS = {
    "step_limit": "step_limit_exceeded",
    "loop_guard": "loop_guard",
    "script_exhausted": "no_termination",
    "policy_error": "policy_error",
}


def verify_episode(task: Any, episode: EpisodeResult) -> VerificationResult:
    """Verify one episode against its task's expected outcome.

    Args:
        task: a :class:`toolforge.tasks.Task` (needs ``expected``, ``family``,
            ``tool_families``).
        episode: finished episode from :class:`toolforge.executor.Executor`.

    Raises:
        VerifierError: if the task has no expected outcome or a check spec is
            malformed — evaluation must fail loudly, never silently pass.
    """
    expected = getattr(task, "expected", None)
    if expected is None:
        raise VerifierError(f"Task {task.task_id} has no expected outcome; cannot verify")
    if episode.task_id != task.task_id:
        raise VerifierError(f"Episode/task mismatch: {episode.task_id} vs {task.task_id}")

    categories: list[str] = []
    checks: list[CheckResult] = []

    # 1) Termination structure.
    if episode.termination in _BAD_TERMINATIONS:
        categories.append(_BAD_TERMINATIONS[episode.termination])
    terminal = expected.terminal_action
    if terminal == "clarify":
        if episode.termination == "final_answer":
            categories.append("missing_clarification")
        elif episode.termination not in ("clarify",) and episode.termination not in _BAD_TERMINATIONS:
            categories.append("missing_clarification")
    elif terminal == "final_answer":
        if episode.termination == "clarify":
            categories.append("unnecessary_clarification")
        elif episode.termination != "final_answer" and episode.termination not in _BAD_TERMINATIONS:
            categories.append("no_termination")
    elif terminal == "any":
        if episode.termination not in ("final_answer", "clarify") \
                and episode.termination not in _BAD_TERMINATIONS:
            categories.append("no_termination")
    else:
        raise VerifierError(f"Unknown terminal_action {terminal!r} on task {task.task_id}")

    # 2) Forbidden global side effects.
    forbidden_changes: list[str] = []
    if expected.forbid_state_changes:
        for family in task.tool_families:
            ignore = frozenset(episode.volatile_keys.get(family, ()))
            changes = diff_states(episode.initial_states.get(family, {}),
                                  episode.final_states.get(family, {}),
                                  path=family, ignore_keys=ignore)
            forbidden_changes.extend(changes)
        if forbidden_changes:
            categories.append("forbidden_state_change")

    # 3) Declarative checks.
    for spec in expected.checks:
        check_result = evaluate_check(spec, episode, task.family)
        checks.append(check_result)
        if not check_result.passed and check_result.category:
            categories.append(check_result.category)

    # 4) Optional invalid-call budget.
    if expected.max_invalid_calls is not None and episode.invalid_call_count > expected.max_invalid_calls:
        categories.append("too_many_invalid_calls")

    # De-dup while preserving order (a category may arise from several checks).
    seen: set[str] = set()
    ordered = [c for c in categories if not (c in seen or seen.add(c))]

    stats = {
        "termination": episode.termination,
        "tool_call_count": episode.tool_call_count,
        "invalid_call_count": episode.invalid_call_count,
        "parse_error_count": episode.parse_error_count,
        "turns": len(episode.trace),
        "policy_seconds": round(episode.policy_seconds, 6),
        "env_seconds": round(episode.env_seconds, 6),
    }
    return VerificationResult(
        task_id=task.task_id,
        success=not ordered,
        failure_categories=ordered,
        checks=checks,
        forbidden_changes=forbidden_changes,
        termination=episode.termination,
        stats=stats,
    )
