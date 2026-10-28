"""Metric aggregation over per-task evaluation records.

All formulas are explicit here and mirrored in the README; every number in a
report must trace back to saved prediction records.

Confidence intervals — two distinct kinds, never conflated:

- ``wilson_ci``: binomial interval over the *task sample* of one run.
- ``bootstrap_ci``: percentile bootstrap over per-task 0/1 outcomes of one run
  (``ci_type: "bootstrap_over_tasks"``), seeded and reproducible.
- Run-to-run variability from *repeated training* is a different quantity and
  is computed in Phase 5 from multiple evaluation runs, labeled accordingly.

Rate definitions (denominators matter and are logged):

- ``invalid_json_rate`` = parse-error turns / total assistant turns.
- ``invalid_tool_call_rate`` = invalid calls (parse errors on tool-call turns +
  unknown tools + schema violations) / all tool-call attempts (invalid + executed).
- ``argument_validation_failure_rate`` = turns with schema-violation issues /
  parsed tool-call turns (unknown-tool turns excluded: they have no valid name
  to validate arguments against).
- ``step_limit_violation_rate`` / ``loop_guard_rate`` = terminations / episodes.
- ``unwanted_side_effect_rate`` = episodes failing the forbidden-state-change
  rule / episodes (only tasks that forbid changes can contribute; the raw
  count is reported alongside).
- ``clarification_*``: precision/recall over tasks whose expected terminal
  action is exactly ``clarify`` or ``final_answer``; tasks accepting either
  (``any``) are excluded because a clarify there is not a false positive.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import Counter
from typing import Any, Iterable, Sequence

from toolforge.schemas import ToolForgeError

Z_95 = 1.959963984540054


class MetricsError(ToolForgeError):
    """Raised when records are missing fields needed for aggregation."""


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval; ``None`` when n == 0 (no data, no interval)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = center - spread, center + spread
    # Snap float epsilon to the exact bounds (p=0 -> lo=0, p=1 -> hi=1) so
    # downstream consumers (e.g. matplotlib error bars) never see -1e-17.
    lo = 0.0 if lo < 1e-12 else lo
    hi = 1.0 if hi > 1 - 1e-12 else hi
    return (max(0.0, lo), min(1.0, hi))


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, iterations: int = 10_000,
                 confidence: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean over task-level values."""
    n = len(values)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations)]
    means.sort()
    lo_idx = int((1 - confidence) / 2 * iterations)
    hi_idx = min(iterations - 1, int((1 + confidence) / 2 * iterations))
    return (means[lo_idx], means[hi_idx])


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy-compatible); None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(values, 0.5), "p95": percentile(values, 0.95)}


# --------------------------------------------------------------------------
# Group aggregation
# --------------------------------------------------------------------------

def _group_rates(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r[key]), []).append(r)
    out = {}
    for name in sorted(groups):
        rs = groups[name]
        successes = sum(1 for r in rs if r["success"])
        ci = wilson_ci(successes, len(rs))
        out[name] = {"n": len(rs), "successes": successes,
                     "success_rate": successes / len(rs),
                     "wilson_ci_low": ci[0] if ci else None,
                     "wilson_ci_high": ci[1] if ci else None}
    return out


def clarification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision/recall/F1 for clarification behavior on labeled tasks."""
    labeled = [r for r in records if r["expected_terminal_action"] in ("clarify", "final_answer")]
    tp = sum(1 for r in labeled if r["expected_clarify"] and r["predicted_clarify"])
    fp = sum(1 for r in labeled if not r["expected_clarify"] and r["predicted_clarify"])
    fn = sum(1 for r in labeled if r["expected_clarify"] and not r["predicted_clarify"])
    tn = sum(1 for r in labeled if not r["expected_clarify"] and not r["predicted_clarify"])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and precision + recall > 0 else None)
    return {"n_labeled": len(labeled), "excluded_terminal_any": len(records) - len(labeled),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_negatives": tn, "precision": precision, "recall": recall, "f1": f1}


def _require_fields(records: list[dict[str, Any]]) -> None:
    required = {"task_id", "success", "termination", "failure_categories",
                "tool_call_count", "invalid_call_count", "parse_error_count",
                "turns", "policy_seconds", "env_seconds", "category", "family",
                "difficulty", "split",
                "expected_terminal_action", "expected_clarify", "predicted_clarify"}
    for i, r in enumerate(records):
        missing = required - set(r)
        if missing:
            raise MetricsError(f"Record {i} ({r.get('task_id', '?')}) missing fields: {sorted(missing)}")


# --------------------------------------------------------------------------
# Main summary
# --------------------------------------------------------------------------

def summarize(records: list[dict[str, Any]], *, held_out_families: Iterable[str] = ("crm",),
              bootstrap_seed: int = 0, bootstrap_iterations: int = 10_000,
              run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate per-task records into the report-ready metrics dictionary."""
    if not records:
        raise MetricsError("Cannot summarize zero records; evaluation produced no predictions")
    _require_fields(records)
    held_out = set(held_out_families)
    n = len(records)
    successes = sum(1 for r in records if r["success"])
    outcomes = [1.0 if r["success"] else 0.0 for r in records]
    wilson = wilson_ci(successes, n)
    boot = bootstrap_ci(outcomes, seed=bootstrap_seed, iterations=bootstrap_iterations)

    # Per-task mean success (matters when repeat > 1).
    per_task: dict[str, list[float]] = {}
    for r in records:
        per_task.setdefault(r["task_id"], []).append(1.0 if r["success"] else 0.0)
    task_means = [sum(v) / len(v) for v in per_task.values()]
    task_mean_variability = {
        "tasks_with_mixed_outcomes": sum(1 for v in per_task.values() if len(set(v)) > 1),
        "runs_per_task": dict(Counter(len(v) for v in per_task.values())),
    }

    total_turns = sum(r["turns"] for r in records)
    total_tool_calls = sum(r["tool_call_count"] for r in records)
    total_invalid = sum(r["invalid_call_count"] for r in records)
    total_parse_errors = sum(r["parse_error_count"] for r in records)
    arg_fail_steps = sum(r.get("arg_validation_failure_steps", 0) for r in records)
    tool_call_turns = sum(r.get("tool_call_action_steps", 0) for r in records)
    parsed_tool_call_turns = sum(r.get("parsed_tool_call_steps", 0) for r in records)
    terminations = Counter(r["termination"] for r in records)
    failure_categories = Counter(c for r in records for c in r["failure_categories"])
    side_effect_episodes = sum(1 for r in records
                               if "forbidden_state_change" in r["failure_categories"])

    held = [r for r in records if r["family"] in held_out]
    seen = [r for r in records if r["family"] not in held_out]

    def subset_rate(rs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rs:
            return {"n": 0, "success_rate": None, "wilson_ci_low": None, "wilson_ci_high": None}
        k = sum(1 for r in rs if r["success"])
        ci = wilson_ci(k, len(rs))
        return {"n": len(rs), "successes": k, "success_rate": k / len(rs),
                "wilson_ci_low": ci[0] if ci else None, "wilson_ci_high": ci[1] if ci else None}

    multi_step = [r for r in records if r["category"] == "multi_step_workflow"]
    workflow_like = [r for r in records if r["category"] in
                     ("multi_step_workflow", "dependent_actions", "indirect_reference")]

    tokens_prompt = [r["prompt_tokens"] for r in records if r.get("prompt_tokens") is not None]
    tokens_completion = [r["completion_tokens"] for r in records if r.get("completion_tokens") is not None]
    ttft = [r["ttft_seconds"] for r in records if r.get("ttft_seconds") is not None]

    success_latency = latency_summary([r["policy_seconds"] for r in records if r["success"]])
    success_wall = latency_summary([r["policy_seconds"] + r["env_seconds"]
                                    for r in records if r["success"]])

    summary: dict[str, Any] = {
        "n_records": n,
        "n_tasks": len(per_task),
        "success_rate": successes / n,
        "successes": successes,
        "wilson_ci": {"low": wilson[0], "high": wilson[1], "z": Z_95} if wilson else None,
        "bootstrap_ci": {"low": boot[0], "high": boot[1], "seed": bootstrap_seed,
                         "iterations": bootstrap_iterations,
                         "ci_type": "bootstrap_over_tasks"} if boot else None,
        "mean_per_task_success": sum(task_means) / len(task_means),
        "task_mean_variability": task_mean_variability,
        "success_by_category": _group_rates(records, "category"),
        "success_by_family": _group_rates(records, "family"),
        "success_by_difficulty": _group_rates(records, "difficulty"),
        "success_by_split": _group_rates(records, "split"),
        "held_out_family_success": {**subset_rate(held), "families": sorted(held_out)},
        "seen_family_success": subset_rate(seen),
        "multi_step_success": subset_rate(multi_step),
        "workflow_like_success": subset_rate(workflow_like),
        "clarification": clarification_metrics(records),
        "invalid_json_rate": total_parse_errors / total_turns if total_turns else 0.0,
        "invalid_tool_call_rate": (total_invalid / (total_invalid + total_tool_calls)
                                   if (total_invalid + total_tool_calls) else 0.0),
        "argument_validation_failure_rate": (arg_fail_steps / parsed_tool_call_turns
                                             if parsed_tool_call_turns else 0.0),
        "rate_denominators": {
            "assistant_turns": total_turns,
            "tool_call_attempts_invalid_plus_executed": total_invalid + total_tool_calls,
            "parsed_tool_call_turns": parsed_tool_call_turns,
            "tool_call_action_steps_incl_unknown": tool_call_turns,
        },
        "step_limit_violation_rate": terminations.get("step_limit", 0) / n,
        "loop_guard_rate": terminations.get("loop_guard", 0) / n,
        "termination_counts": dict(terminations),
        "unwanted_side_effect_rate": side_effect_episodes / n,
        "failure_category_counts": dict(failure_categories.most_common()),
        "avg_tool_calls_per_task": total_tool_calls / n,
        "avg_turns_per_task": total_turns / n,
        "avg_completion_chars_per_turn": (
            sum(r.get("assistant_output_chars", 0) for r in records) / total_turns
            if total_turns else 0.0),
        "avg_final_output_chars": (
            sum(r.get("final_output_chars") or 0 for r in records) / n),
        "latency": {
            "policy_seconds_per_task": latency_summary([r["policy_seconds"] for r in records]),
            "env_seconds_per_task": latency_summary([r["env_seconds"] for r in records]),
            "wall_seconds_per_task": latency_summary(
                [r["policy_seconds"] + r["env_seconds"] for r in records]),
            "time_to_successful_completion_policy_seconds": success_latency,
            "time_to_successful_completion_wall_seconds": success_wall,
        },
        "tokens": {
            "records_with_token_stats": len(tokens_prompt),
            "prompt_tokens_total": sum(tokens_prompt) if tokens_prompt else None,
            "completion_tokens_total": sum(tokens_completion) if tokens_completion else None,
            "prompt_tokens_mean_per_task": (sum(tokens_prompt) / len(tokens_prompt)
                                            if tokens_prompt else None),
            "completion_tokens_mean_per_task": (sum(tokens_completion) / len(tokens_completion)
                                                if tokens_completion else None),
            "ttft_seconds": latency_summary(ttft) if ttft else None,
        },
        "run_metadata": run_metadata or {},
    }
    return summary


# --------------------------------------------------------------------------
# Flat CSV view (long format: one row per scope/key metric)
# --------------------------------------------------------------------------

def summary_to_csv(summary: dict[str, Any]) -> str:
    """Flatten the headline + group metrics into a tidy CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["scope", "key", "n", "successes", "rate", "wilson_ci_low", "wilson_ci_high"])

    def rate_row(scope: str, key: str, d: dict[str, Any]) -> None:
        writer.writerow([scope, key, d.get("n"), d.get("successes"),
                         _fmt(d.get("success_rate")), _fmt(d.get("wilson_ci_low")),
                         _fmt(d.get("wilson_ci_high"))])

    rate_row("overall", "all_tasks",
             {"n": summary["n_records"], "successes": summary["successes"],
              "success_rate": summary["success_rate"],
              "wilson_ci_low": (summary["wilson_ci"] or {}).get("low"),
              "wilson_ci_high": (summary["wilson_ci"] or {}).get("high")})
    for scope in ("success_by_category", "success_by_family", "success_by_difficulty",
                  "success_by_split"):
        for key, d in summary[scope].items():
            rate_row(scope, key, d)
    rate_row("subset", "held_out_family", summary["held_out_family_success"])
    rate_row("subset", "seen_family", summary["seen_family_success"])
    rate_row("subset", "multi_step", summary["multi_step_success"])
    rate_row("subset", "workflow_like", summary["workflow_like_success"])

    writer.writerow([])
    writer.writerow(["metric", "value"])
    scalars = {
        "invalid_json_rate": summary["invalid_json_rate"],
        "invalid_tool_call_rate": summary["invalid_tool_call_rate"],
        "argument_validation_failure_rate": summary["argument_validation_failure_rate"],
        "step_limit_violation_rate": summary["step_limit_violation_rate"],
        "loop_guard_rate": summary["loop_guard_rate"],
        "unwanted_side_effect_rate": summary["unwanted_side_effect_rate"],
        "avg_tool_calls_per_task": summary["avg_tool_calls_per_task"],
        "avg_turns_per_task": summary["avg_turns_per_task"],
        "avg_completion_chars_per_turn": summary["avg_completion_chars_per_turn"],
        "clarification_precision": summary["clarification"]["precision"],
        "clarification_recall": summary["clarification"]["recall"],
        "clarification_f1": summary["clarification"]["f1"],
        "bootstrap_ci_low": (summary["bootstrap_ci"] or {}).get("low"),
        "bootstrap_ci_high": (summary["bootstrap_ci"] or {}).get("high"),
    }
    for key, value in scalars.items():
        writer.writerow([key, _fmt(value)])
    return buf.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(summary: dict[str, Any], json_path: str, csv_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(summary_to_csv(summary))
