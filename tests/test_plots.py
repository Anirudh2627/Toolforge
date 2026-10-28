"""Plot tests: figures render from provided data only, refuse missing data,
and illustrative mode is loudly labeled. Synthetic tables here are TEST
FIXTURES for the plotting code, never published results.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from toolforge.plots import (  # noqa: E402
    PlotDataError,
    illustrative_table,
    plot_all,
    plot_behavior_rates,
    plot_category_success,
    plot_latency_vs_success,
    plot_overall_comparison,
    plot_rl_reward_vs_accuracy,
    plot_seen_vs_heldout,
)
from toolforge.tasks import CATEGORIES  # noqa: E402


def make_table() -> dict:
    table = {}
    for i, (label, rate) in enumerate([("base", 0.20), ("sft", 0.50), ("sft-grpo", 0.58)]):
        table[label] = {
            "model": label, "split": "test", "grammar_version": "tf-gram-v1",
            "n_tasks": 59, "n_records": 59, "success_rate": rate,
            "wilson_ci": {"low": rate - 0.08, "high": rate + 0.08},
            "seen": {"success_rate": rate + 0.05},
            "held_out": {"success_rate": max(0.0, rate - 0.10)},
            "by_category": {c: min(1.0, rate + (j % 5 - 2) * 0.04)
                            for j, c in enumerate(CATEGORIES)},
            "invalid_json_rate": 0.2 - 0.05 * i,
            "invalid_tool_call_rate": 0.15 - 0.04 * i,
            "argument_validation_failure_rate": 0.1 - 0.03 * i,
            "step_limit_violation_rate": 0.3 - 0.1 * i,
            "unwanted_side_effect_rate": 0.05,
            "policy_seconds_median": 4.0 + 0.5 * i,
            "policy_seconds_p95": 9.0 + i,
            "clarification": {"precision": 0.5, "recall": 0.4, "f1": 0.44},
            "failure_category_counts": {"state_check_failed": 3},
        }
    return table


def _is_png(path) -> bool:
    return path.exists() and path.read_bytes()[:4] == b"\x89PNG" and path.stat().st_size > 2000


class TestPlots:
    def test_overall(self, tmp_path):
        out = plot_overall_comparison(make_table(), tmp_path / "overall.png")
        assert _is_png(out)

    def test_categories(self, tmp_path):
        assert _is_png(plot_category_success(make_table(), tmp_path / "cat.png"))

    def test_seen_vs_heldout(self, tmp_path):
        assert _is_png(plot_seen_vs_heldout(make_table(), tmp_path / "heldout.png"))

    def test_behavior_rates(self, tmp_path):
        assert _is_png(plot_behavior_rates(make_table(), tmp_path / "behavior.png"))

    def test_latency_scatter(self, tmp_path):
        assert _is_png(plot_latency_vs_success(make_table(), tmp_path / "latency.png"))

    def test_missing_data_raises_never_fabricates(self, tmp_path):
        with pytest.raises(PlotDataError):
            plot_overall_comparison({}, tmp_path / "x.png")
        broken = make_table()
        del broken["sft"]["policy_seconds_median"]
        broken["sft"]["policy_seconds_median"] = None
        with pytest.raises(PlotDataError):
            plot_latency_vs_success(broken, tmp_path / "y.png")
        no_cats = {k: {**v, "by_category": {}} for k, v in make_table().items()}
        with pytest.raises(PlotDataError):
            plot_category_success(no_cats, tmp_path / "z.png")

    def test_rl_curve(self, tmp_path):
        curve = {"iterations": [0, 1, 2, 3, 4],
                 "window_reward_mean": [0.1, 0.25, 0.3, 0.45, 0.5],
                 "eval_points": [(1, 0.2), (3, 0.35)],
                 "divergence_iterations": [4]}
        assert _is_png(plot_rl_reward_vs_accuracy(curve, "grpo-run", tmp_path / "rl.png"))
        # divergence-only history without evals still renders
        curve2 = {**curve, "eval_points": []}
        assert _is_png(plot_rl_reward_vs_accuracy(curve2, "g2", tmp_path / "rl2.png"))
        with pytest.raises(PlotDataError):
            plot_rl_reward_vs_accuracy({"iterations": [], "window_reward_mean": []},
                                       "empty", tmp_path / "rl3.png")

    def test_plot_all(self, tmp_path):
        written = plot_all(make_table(), tmp_path / "figs")
        assert len(written) >= 5
        assert all(_is_png(p) for p in written)

    def test_illustrative_is_labeled_and_prefixed(self, tmp_path):
        table = illustrative_table()
        assert all(l.startswith("ILLUSTRATIVE") for l in table)
        written = plot_all(table, tmp_path / "prev", illustrative=True)
        assert written and all(p.name.startswith("illustrative_") for p in written)
        assert all(_is_png(p) for p in written)
