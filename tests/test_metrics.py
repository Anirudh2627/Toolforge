"""Tests for metric aggregation, confidence intervals, and CSV/JSON output."""

from __future__ import annotations

import json

import pytest

from toolforge.metrics import (
    MetricsError,
    bootstrap_ci,
    clarification_metrics,
    latency_summary,
    percentile,
    summarize,
    summary_to_csv,
    wilson_ci,
    write_summary,
)


def make_record(task_id: str, *, success: bool, category: str = "single_step_selection",
                family: str = "calendar", difficulty: str = "easy", split: str = "test",
                termination: str = "final_answer", failure_categories=None,
                expected_terminal: str = "final_answer", predicted_clarify: bool = False,
                turns: int = 2, tool_calls: int = 1, invalid: int = 0, parse_errors: int = 0,
                arg_fail_steps: int = 0, parsed_tool_steps: int = 1, tool_action_steps: int = 1,
                policy_seconds: float = 1.0, env_seconds: float = 0.01,
                run_index: int = 0, **extra) -> dict:
    record = {
        "task_id": task_id, "run_index": run_index, "sampling_seed": 0,
        "policy_id": "test", "split": split, "family": family, "category": category,
        "difficulty": difficulty, "success": success,
        "failure_categories": failure_categories or [], "termination": termination,
        "expected_terminal_action": expected_terminal,
        "expected_clarify": expected_terminal == "clarify",
        "predicted_clarify": predicted_clarify,
        "turns": turns, "tool_call_count": tool_calls, "invalid_call_count": invalid,
        "parse_error_count": parse_errors,
        "arg_validation_failure_steps": arg_fail_steps,
        "parsed_tool_call_steps": parsed_tool_steps,
        "tool_call_action_steps": tool_action_steps,
        "assistant_output_chars": turns * 40, "final_output_chars": 30,
        "policy_seconds": policy_seconds, "env_seconds": env_seconds,
        "prompt_tokens": None, "completion_tokens": None, "ttft_seconds": None,
    }
    record.update(extra)
    return record


class TestPrimitives:
    def test_wilson_known_value(self):
        lo, hi = wilson_ci(50, 100)
        assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3

    def test_wilson_edges(self):
        assert wilson_ci(0, 10)[0] == 0.0
        assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
        assert wilson_ci(0, 0) is None

    def test_bootstrap_deterministic_and_contains_mean(self):
        values = [1.0] * 30 + [0.0] * 70
        ci1 = bootstrap_ci(values, seed=42, iterations=2000)
        ci2 = bootstrap_ci(values, seed=42, iterations=2000)
        assert ci1 == ci2
        assert ci1[0] <= 0.3 <= ci1[1]
        assert bootstrap_ci([], seed=1) is None

    def test_bootstrap_interval_has_width(self):
        values = [float(i % 2) for i in range(100)]
        lo, hi = bootstrap_ci(values, seed=1, iterations=2000)
        assert hi > lo  # sampling variability is visible in the interval

    def test_percentile(self):
        assert percentile([], 0.5) is None
        assert percentile([5], 0.95) == 5
        assert percentile(list(range(1, 11)), 0.5) == 5.5
        assert percentile(list(range(1, 11)), 0.95) == pytest.approx(9.55)

    def test_latency_summary(self):
        empty = latency_summary([])
        assert empty["n"] == 0 and empty["median"] is None
        s = latency_summary([1.0, 2.0, 3.0, 10.0])
        assert s["n"] == 4 and s["mean"] == 4.0 and s["median"] == 2.5


class TestClarificationMetrics:
    def test_prf_and_any_exclusion(self):
        records = [
            make_record("a", success=True, expected_terminal="clarify", predicted_clarify=True),
            make_record("b", success=False, expected_terminal="clarify", predicted_clarify=False),
            make_record("c", success=False, expected_terminal="final_answer", predicted_clarify=True),
            make_record("d", success=True, expected_terminal="final_answer", predicted_clarify=False),
            make_record("e", success=True, expected_terminal="any", predicted_clarify=True),
        ]
        m = clarification_metrics(records)
        assert m["n_labeled"] == 4 and m["excluded_terminal_any"] == 1
        assert (m["true_positives"], m["false_positives"], m["false_negatives"],
                m["true_negatives"]) == (1, 1, 1, 1)
        assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5

    def test_empty_denominators_are_none(self):
        m = clarification_metrics([make_record("a", success=True)])
        assert m["precision"] is None and m["recall"] is None and m["f1"] is None


class TestSummarize:
    def test_headline_numbers(self):
        records = [
            make_record("t1", success=True),
            make_record("t2", success=False, failure_categories=["state_check_failed"]),
            make_record("t3", success=False, termination="step_limit",
                        failure_categories=["step_limit_exceeded", "forbidden_state_change"]),
            make_record("t4", success=True, family="crm", split="test"),
        ]
        s = summarize(records, bootstrap_iterations=500)
        assert s["n_records"] == 4 and s["success_rate"] == 0.5
        assert s["step_limit_violation_rate"] == 0.25
        assert s["unwanted_side_effect_rate"] == 0.25
        assert s["held_out_family_success"]["n"] == 1
        assert s["held_out_family_success"]["success_rate"] == 1.0
        assert s["seen_family_success"]["n"] == 3
        assert s["success_by_category"]["single_step_selection"]["n"] == 4
        assert s["failure_category_counts"]["state_check_failed"] == 1

    def test_rate_formulas(self):
        # 4 turns total, 1 parse error; 2 tool attempts (1 invalid + 1 executed);
        # of 1 parsed tool_call turn, 1 had schema violations -> rate 1.0.
        records = [make_record("t1", success=False, turns=3, parse_errors=1, invalid=1,
                               tool_calls=1, arg_fail_steps=1, parsed_tool_steps=1,
                               tool_action_steps=2)]
        s = summarize(records, bootstrap_iterations=100)
        assert s["invalid_json_rate"] == pytest.approx(1 / 3)
        assert s["invalid_tool_call_rate"] == pytest.approx(0.5)
        assert s["argument_validation_failure_rate"] == pytest.approx(1.0)
        assert s["rate_denominators"]["assistant_turns"] == 3

    def test_repeat_runs_per_task_mean(self):
        records = [
            make_record("t1", success=True, run_index=0),
            make_record("t1", success=False, run_index=1),
            make_record("t2", success=True, run_index=0),
            make_record("t2", success=True, run_index=1),
        ]
        s = summarize(records, bootstrap_iterations=100)
        assert s["n_records"] == 4 and s["n_tasks"] == 2
        assert s["mean_per_task_success"] == pytest.approx((0.5 + 1.0) / 2)
        assert s["task_mean_variability"]["tasks_with_mixed_outcomes"] == 1

    def test_empty_and_missing_fields(self):
        with pytest.raises(MetricsError):
            summarize([])
        broken = make_record("t1", success=True)
        del broken["category"]
        with pytest.raises(MetricsError, match="category"):
            summarize([broken])

    def test_bootstrap_ci_labeled(self):
        s = summarize([make_record("t1", success=True)], bootstrap_iterations=100)
        assert s["bootstrap_ci"]["ci_type"] == "bootstrap_over_tasks"


class TestOutputs:
    def test_csv_contains_scopes(self):
        s = summarize([make_record("t1", success=True), make_record("t2", success=False)],
                      bootstrap_iterations=100)
        csv_text = summary_to_csv(s)
        assert "scope,key,n,successes,rate" in csv_text
        assert "overall,all_tasks" in csv_text
        assert "success_by_family,calendar" in csv_text
        assert "subset,held_out_family" in csv_text
        assert "invalid_json_rate" in csv_text

    def test_write_summary_roundtrip(self, tmp_path):
        s = summarize([make_record("t1", success=True)], bootstrap_iterations=100)
        write_summary(s, str(tmp_path / "m.json"), str(tmp_path / "m.csv"))
        loaded = json.loads((tmp_path / "m.json").read_text())
        assert loaded["success_rate"] == 1.0
        assert (tmp_path / "m.csv").read_text().startswith("scope,key")
