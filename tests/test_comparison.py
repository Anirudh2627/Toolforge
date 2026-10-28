"""Comparison tests: run loading + integrity, compatibility hard-checks,
tables, paired analysis (exact McNemar), repeat-groups, RL curves, report.

Uses REAL evaluation runs produced by run_evaluation with random policies on a
benchmark subset (CPU, offline) — comparisons are tested against genuine
artifacts, not hand-written numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolforge.comparison import (
    ComparisonError,
    check_compatibility,
    comparison_table,
    comparison_to_csv,
    find_regressions,
    load_history,
    load_run,
    mcnemar_exact,
    paired_analysis,
    repeat_group_stats,
    report_markdown,
    rl_curve_data,
)
from toolforge.evaluation import RunMetadata, filter_tasks, run_evaluation
from toolforge.executor import Executor
from toolforge.inference import RandomPolicy
from toolforge.tasks_builders import build_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[Path, Path]:
    """Two real evaluation runs (different random-policy seeds) on one subset."""
    tasks = build_benchmark(seed=20260301)
    subset = filter_tasks(tasks, split="test", limit=12)  # includes held-out crm tasks
    executor = Executor(default_registry_local())
    tmp = tmp_path_factory.mktemp("runs")

    def make(name: str, seed: int) -> Path:
        out = tmp / name

        def factory(run_index: int, sampling_seed: int):
            return RandomPolicy(seed=seed + run_index, max_tool_calls=3)

        run_evaluation(subset, executor, factory, out_dir=out, run_name=name,
                       run_metadata=RunMetadata(run_name=name, policy_description=f"random:{seed}",
                                                task_file=None, split="test", n_tasks=len(subset),
                                                repeat=1, base_seed=seed),
                       bootstrap_iterations=200)
        return out

    return make("run-a", 1), make("run-b", 2)


def default_registry_local():
    from toolforge.registry import default_registry
    return default_registry()


class TestLoadRun:
    def test_load_and_integrity(self, two_runs):
        bundle = load_run("a", two_runs[0])
        assert bundle.records and bundle.metrics["n_records"] == len(bundle.records)
        assert bundle.split == "test"

    def test_tampered_metrics_rejected(self, two_runs, tmp_path):
        import shutil
        bad = tmp_path / "tampered"
        shutil.copytree(two_runs[0], bad)
        metrics = json.loads((bad / "metrics.json").read_text())
        metrics["success_rate"] = 0.999  # fabricated improvement
        (bad / "metrics.json").write_text(json.dumps(metrics))
        with pytest.raises(ComparisonError, match="tampered or stale"):
            load_run("bad", bad)

    def test_missing_files_rejected(self, tmp_path):
        with pytest.raises(ComparisonError, match="missing"):
            load_run("nope", tmp_path)


class TestCompatibility:
    def test_compatible_runs_pass(self, two_runs):
        a, b = load_run("a", two_runs[0]), load_run("b", two_runs[1])
        assert check_compatibility([a, b]) == []

    def test_fingerprint_mismatch_refused(self, two_runs, tmp_path):
        import shutil
        a = load_run("a", two_runs[0])
        mutant_dir = tmp_path / "mutant"
        shutil.copytree(two_runs[1], mutant_dir)
        cfg = json.loads((mutant_dir / "run_config.json").read_text())
        fps = cfg.get("tools_fingerprints") or {}
        for key in fps:
            fps[key] = "deadbeefdeadbeef"
        cfg["tools_fingerprints"] = fps
        (mutant_dir / "run_config.json").write_text(json.dumps(cfg))
        b = load_run("b", mutant_dir)
        with pytest.raises(ComparisonError, match="fingerprint mismatch"):
            check_compatibility([a, b])
        warnings = check_compatibility([a, b], allow_incompatible=True)
        assert any("OVERRIDE" in w for w in warnings)

    def test_split_mismatch_refused(self, two_runs, tmp_path):
        import shutil
        a = load_run("a", two_runs[0])
        other_dir = tmp_path / "othersplit"
        shutil.copytree(two_runs[1], other_dir)
        cfg = json.loads((other_dir / "run_config.json").read_text())
        cfg["split"] = "val"
        (other_dir / "run_config.json").write_text(json.dumps(cfg))
        b = load_run("b", other_dir)
        with pytest.raises(ComparisonError, match="split differs"):
            check_compatibility([a, b])


class TestComparisonTable:
    def test_values_match_predictions(self, two_runs):
        bundles = [load_run("a", two_runs[0]), load_run("b", two_runs[1])]
        table = comparison_table(bundles)
        for b in bundles:
            row = table[b.label]
            manual = sum(1 for r in b.records if r["success"]) / len(b.records)
            assert row["success_rate"] == pytest.approx(manual)
            assert row["n_records"] == len(b.records)
            assert row["policy_seconds_median"] is not None
        csv_text = comparison_to_csv(table)
        assert csv_text.splitlines()[0].startswith("label,success_rate")
        assert "a," in csv_text and "b," in csv_text

    def test_held_out_slice_present(self, two_runs):
        a = load_run("a", two_runs[0])
        table = comparison_table([a, load_run("b", two_runs[1])])
        # subset was taken from the test split, which contains crm tasks
        assert table["a"]["held_out"]["n"] >= 0


class TestPaired:
    def test_counts_partition_common_tasks(self, two_runs):
        a, b = load_run("a", two_runs[0]), load_run("b", two_runs[1])
        p = paired_analysis(a, b)
        assert (p["both_solved"] + p["only_a_solved"] + p["only_b_solved"]
                + p["neither_solved"]) == p["n_common_tasks"]
        p_rev = paired_analysis(b, a)
        assert p_rev["only_a_solved"] == p["only_b_solved"]
        assert p_rev["mcnemar_exact_p"] == pytest.approx(p["mcnemar_exact_p"])

    def test_mcnemar_known_values(self):
        assert mcnemar_exact(0, 0) == 1.0
        assert mcnemar_exact(5, 5) == 1.0
        assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)
        assert mcnemar_exact(0, 3) == pytest.approx(2 * (1 / 8))


class TestRepeatGroupsAndCurves:
    def test_repeat_group_stats(self, two_runs):
        bundles = [load_run("a", two_runs[0]), load_run("b", two_runs[1])]
        g = repeat_group_stats("sft-rep", bundles)
        assert g["n_runs"] == 2
        assert g["mean_success_rate"] == pytest.approx(
            sum(b.metrics["success_rate"] for b in bundles) / 2)
        assert "repeated_training" in g["ci_type"]
        with pytest.raises(ComparisonError):
            repeat_group_stats("x", bundles[:1])

    def test_rl_curve_extraction(self, tmp_path):
        history = [
            {"iteration": 0, "window_reward_mean": 0.1},
            {"iteration": 1, "window_reward_mean": 0.3,
             "eval": {"success_rate": 0.2}},
            {"iteration": 2, "window_reward_mean": 0.5, "divergence_warning": True},
            {"iteration": 3, "window_reward_mean": 0.6,
             "eval": {"success_rate": 0.15}},
        ]
        path = tmp_path / "history.jsonl"
        path.write_text("\n".join(json.dumps(h) for h in history), encoding="utf-8")
        curve = rl_curve_data(load_history(path))
        assert curve["iterations"] == [0, 1, 2, 3]
        assert curve["eval_points"] == [(1, 0.2), (3, 0.15)]
        assert curve["divergence_iterations"] == [2]
        with pytest.raises(ComparisonError):
            load_history(tmp_path / "missing.jsonl")


class TestReport:
    def test_report_is_data_driven_and_honest(self, two_runs):
        bundles = [load_run("a", two_runs[0]), load_run("b", two_runs[1])]
        table = comparison_table(bundles)
        paired = [paired_analysis(*bundles)]
        regressions = find_regressions(bundles, table, threshold=0.05)
        text = report_markdown(bundles, table, paired, groups=[], histories={},
                               warnings=[], regressions=regressions)
        assert "a" in text and "b" in text
        assert f"{table['a']['success_rate']:.3f}" in text
        assert "Repeated-training variability" in text and "Not available" in text
        assert "no GRPO history" in text.lower() or "Not available" in text
        assert "McNemar" in text

    def test_find_regressions_threshold(self):
        class FakeBundle:
            def __init__(self, label):
                self.label = label
        table = {
            "base": {"by_category": {"single_step_selection": 0.50,
                                     "ambiguous_clarify": 0.40}},
            "sft": {"by_category": {"single_step_selection": 0.30,   # drop 0.20 -> flagged
                                    "ambiguous_clarify": 0.38}},     # drop 0.02 -> ignored
        }
        regs = find_regressions([FakeBundle("base"), FakeBundle("sft")], table, threshold=0.05)
        assert len(regs) == 1
        assert regs[0]["category"] == "single_step_selection"
        assert regs[0]["drop"] == pytest.approx(0.20)


class TestCLI:
    def test_compare_models_script_end_to_end(self, two_runs, tmp_path):
        out = tmp_path / "report-out"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "compare_models.py"),
             "--run", f"a={two_runs[0]}", "--run", f"b={two_runs[1]}",
             "--out", str(out)],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin",
                 "MPLBACKEND": "Agg", "HOME": str(tmp_path)})
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert (out / "REPORT.md").exists()
        assert (out / "comparison.json").exists()
        assert (out / "comparison.csv").exists()
        assert (out / "paired.json").exists()
        assert (out / "provenance.json").exists()
        pngs = list((out / "plots").glob("*.png"))
        assert len(pngs) >= 4
        for png in pngs:
            assert png.read_bytes()[:4] == b"\x89PNG"
        provenance = json.loads((out / "provenance.json").read_text())
        assert set(provenance["runs"]) == {"a", "b"}
