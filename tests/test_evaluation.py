"""End-to-end tests for the evaluation runner: preflight, records, resume,
repeat runs, policy-error handling, and metrics recomputation.
"""

from __future__ import annotations

import json

import pytest

from toolforge.evaluation import (
    EvaluationError,
    RunMetadata,
    filter_tasks,
    load_completed,
    load_predictions,
    preflight_tasks,
    reaggregate,
    run_evaluation,
)
from toolforge.executor import ScriptedPolicy
from toolforge.inference import RandomPolicy
from toolforge.schemas import GRAMMAR_VERSION
from toolforge.tasks import ExpectedOutcome, Task
from toolforge.tasks_builders import build_benchmark

SUBSET_IDS = ["tf-cal-ss-001", "tf-cal-amb-001", "tf-cal-imp-001", "tf-eml-ss-001",
              "tf-db-ss-001", "tf-td-ss-001", "tf-fs-ss-003", "tf-crm-ss-001"]


@pytest.fixture(scope="module")
def benchmark() -> list[Task]:
    return build_benchmark(seed=20260301)


@pytest.fixture(scope="module")
def subset(benchmark) -> list[Task]:
    by_id = {t.task_id: t for t in benchmark}
    missing = [i for i in SUBSET_IDS if i not in by_id]
    assert not missing, f"benchmark task ids changed: {missing}"
    return [by_id[i] for i in SUBSET_IDS]


def random_factory(run_index: int, sampling_seed: int) -> RandomPolicy:
    return RandomPolicy(seed=100 + run_index)


def metadata(name: str = "test-run", n: int = 8) -> RunMetadata:
    return RunMetadata(run_name=name, policy_description="random:test", task_file=None,
                       split="mixed", n_tasks=n, repeat=1, base_seed=0)


class TestPreflight:
    def test_full_benchmark_preflights(self, benchmark, executor):
        info = preflight_tasks(benchmark, executor)
        assert info["n_tasks"] == len(benchmark)
        assert info["tools_fingerprints"]  # one fingerprint per family-set

    def test_subset_preflight(self, subset, executor):
        preflight_tasks(subset, executor)

    def test_missing_expected_fails_loudly(self, subset, executor):
        broken = Task.from_dict(subset[0].to_dict())
        broken.expected = None
        with pytest.raises(EvaluationError, match="no expected outcome"):
            preflight_tasks([broken], executor)

    def test_bad_check_spec_fails_loudly(self, subset, executor):
        broken = Task.from_dict(subset[0].to_dict())
        broken.expected = ExpectedOutcome(checks=[{"check": "record_exists",
                                                   "collection": "ghosts"}])
        with pytest.raises(EvaluationError, match="cannot be verified"):
            preflight_tasks([broken], executor)

    def test_bad_fixture_fails_loudly(self, subset, executor):
        broken = Task.from_dict(subset[0].to_dict())
        broken.fixture = {"events": [{"id": "e"}]}  # missing required fields
        with pytest.raises(EvaluationError, match="fixture/family error"):
            preflight_tasks([broken], executor)


class TestFilterTasks:
    def test_filters_and_sort(self, benchmark):
        test_cal = filter_tasks(benchmark, split="test", families=["calendar"])
        assert test_cal and all(t.split == "test" and t.family == "calendar" for t in test_cal)
        assert [t.task_id for t in test_cal] == sorted(t.task_id for t in test_cal)
        by_cat = filter_tasks(benchmark, categories=["ambiguous_clarify"], limit=3)
        assert len(by_cat) == 3 and all(t.category == "ambiguous_clarify" for t in by_cat)
        by_id = filter_tasks(benchmark, task_ids=["tf-cal-ss-001"])
        assert [t.task_id for t in by_id] == ["tf-cal-ss-001"]

    def test_empty_selection_raises(self, benchmark):
        with pytest.raises(EvaluationError, match="zero tasks"):
            filter_tasks(benchmark, families=["nonexistent"])
        with pytest.raises(EvaluationError):
            filter_tasks(benchmark, limit=0)


class TestRunEvaluation:
    def test_writes_all_outputs(self, subset, executor, tmp_path):
        summary = run_evaluation(subset, executor, random_factory,
                                 out_dir=tmp_path / "run1", run_name="run1",
                                 run_metadata=metadata(n=len(subset)),
                                 bootstrap_iterations=200)
        run_dir = tmp_path / "run1"
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "metrics.csv").exists()
        run_cfg = json.loads((run_dir / "run_config.json").read_text())
        assert run_cfg["grammar_version"] == GRAMMAR_VERSION
        assert run_cfg["tools_fingerprints"]
        assert summary["n_records"] == len(subset)

        records = load_predictions(run_dir / "predictions.jsonl")
        assert len(records) == len(subset)
        for r in records:
            assert isinstance(r["success"], bool)
            assert r["policy_id"] == "random:seed=100"
            assert "initial_states" in r and "final_states" in r
            assert r["verdict_checks"] is not None

    def test_resume_skips_completed(self, subset, executor, tmp_path):
        run_dir = tmp_path / "run2"
        run_evaluation(subset, executor, random_factory, out_dir=run_dir, run_name="run2",
                       run_metadata=metadata(n=len(subset)), bootstrap_iterations=200)
        first_lines = (run_dir / "predictions.jsonl").read_text().count("\n")

        def exploding_factory(run_index: int, seed: int):
            raise AssertionError("policy_factory must not be called when everything is done")

        # All (task, run) pairs exist -> the loop body never executes.
        summary = run_evaluation(subset, executor, exploding_factory, out_dir=run_dir,
                                 run_name="run2", run_metadata=metadata(n=len(subset)),
                                 resume=True, bootstrap_iterations=200)
        assert (run_dir / "predictions.jsonl").read_text().count("\n") == first_lines
        assert summary["n_records"] == len(subset)

    def test_resume_disabled_with_existing_file_raises(self, subset, executor, tmp_path):
        run_dir = tmp_path / "run3"
        run_evaluation(subset[:2], executor, random_factory, out_dir=run_dir, run_name="run3",
                       run_metadata=metadata(n=2), bootstrap_iterations=100)
        with pytest.raises(EvaluationError, match="exists"):
            run_evaluation(subset[:2], executor, random_factory, out_dir=run_dir,
                           run_name="run3", run_metadata=metadata(n=2), resume=False,
                           bootstrap_iterations=100)

    def test_repeat_runs(self, subset, executor, tmp_path):
        two = subset[:2]
        run_evaluation(two, executor, random_factory, out_dir=tmp_path / "run4",
                       run_name="run4", run_metadata=metadata(n=2), repeat=2,
                       base_seed=7, bootstrap_iterations=100)
        records = load_predictions(tmp_path / "run4" / "predictions.jsonl")
        assert len(records) == 4
        assert {(r["task_id"], r["run_index"]) for r in records} == \
               {(t.task_id, i) for t in two for i in (0, 1)}
        assert {r["sampling_seed"] for r in records} == {7, 8}

    def test_no_states_option(self, subset, executor, tmp_path):
        run_evaluation(subset[:1], executor, random_factory, out_dir=tmp_path / "run5",
                       run_name="run5", run_metadata=metadata(n=1), include_states=False,
                       bootstrap_iterations=100)
        record = load_predictions(tmp_path / "run5" / "predictions.jsonl")[0]
        assert "initial_states" not in record and "final_states" not in record
        assert record["trace"]  # traces are always kept

    def test_success_path_record(self, benchmark, executor, tmp_path):
        task = next(t for t in benchmark if t.task_id == "tf-cal-ss-001")
        solution = [
            {"action": "tool_call", "tool": "calendar.create_event",
             "arguments": {"title": "Project review", "date": "2026-03-04",
                           "start": "13:00", "end": "14:00"}},
            {"action": "final_answer", "content": "Created 'Project review' on 2026-03-04."},
        ]
        run_evaluation([task], executor, lambda ri, s: ScriptedPolicy(solution),
                       out_dir=tmp_path / "run6", run_name="run6",
                       run_metadata=metadata(n=1), bootstrap_iterations=100)
        record = load_predictions(tmp_path / "run6" / "predictions.jsonl")[0]
        assert record["success"] is True
        assert record["termination"] == "final_answer"
        assert record["tool_call_count"] == 1
        titles = [e["title"] for e in record["final_states"]["calendar"]["events"]]
        assert "Project review" in titles
        assert "Project review" not in \
               [e["title"] for e in record["initial_states"]["calendar"]["events"]]

    def test_policy_error_recorded(self, subset, executor, tmp_path):
        def crashing_policy(ctx):
            raise RuntimeError("simulated CUDA OOM")

        run_evaluation(subset[:1], executor, lambda ri, s: crashing_policy,
                       out_dir=tmp_path / "run7", run_name="run7",
                       run_metadata=metadata(n=1), bootstrap_iterations=100)
        record = load_predictions(tmp_path / "run7" / "predictions.jsonl")[0]
        assert record["success"] is False
        assert record["termination"] == "policy_error"
        assert "policy_error" in record["failure_categories"]
        assert "simulated CUDA OOM" in record["error_message"]

    def test_policy_error_storm_aborts(self, subset, executor, tmp_path):
        def crashing_policy(ctx):
            raise RuntimeError("boom")

        with pytest.raises(EvaluationError, match="Aborting"):
            run_evaluation(subset[:6], executor, lambda ri, s: crashing_policy,
                           out_dir=tmp_path / "run8", run_name="run8",
                           run_metadata=metadata(n=6), bootstrap_iterations=100)

    def test_reaggregate_matches_original(self, subset, executor, tmp_path):
        run_dir = tmp_path / "run9"
        summary = run_evaluation(subset, executor, random_factory, out_dir=run_dir,
                                 run_name="run9", run_metadata=metadata(n=len(subset)),
                                 bootstrap_iterations=200)
        again = reaggregate(run_dir / "predictions.jsonl", tmp_path / "reagg",
                            bootstrap_iterations=200)
        assert again["success_rate"] == summary["success_rate"]
        assert again["success_by_category"] == summary["success_by_category"]

    def test_corrupt_predictions_block_resume(self, subset, executor, tmp_path):
        run_dir = tmp_path / "run10"
        run_dir.mkdir(parents=True)
        (run_dir / "predictions.jsonl").write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="Corrupt line"):
            load_completed(run_dir / "predictions.jsonl")


class TestRunMetadata:
    def test_serializable(self):
        meta = metadata()
        meta.extra = {"argv": ["--policy", "random"]}
        data = meta.to_dict()
        assert json.loads(json.dumps(data))["run_name"] == "test-run"
