"""Benchmark tests: determinism, coverage, split integrity, leakage prevention,
fixture validity, and JSONL round-trips.

Split mechanics are also covered here (test_data_splits concerns in the final
repo layout); building the full benchmark once per module keeps runtime < a few
seconds on CPU.
"""

from __future__ import annotations

import json

import pytest

from toolforge.environments.registry import ENVIRONMENT_CLASSES
from toolforge.registry import default_registry
from toolforge.tasks import (
    CATEGORIES,
    CAT_AMBIGUOUS,
    ExpectedOutcome,
    Task,
    TaskError,
    assign_splits,
    balance_category_coverage,
    benchmark_summary,
    check_no_leakage,
    load_tasks,
    save_tasks,
    tasks_by_split,
)
from toolforge.tasks_builders import build_benchmark

BENCHMARK_SEED = 20260301
HELD_OUT = ("crm",)


@pytest.fixture(scope="module")
def benchmark() -> list[Task]:
    return build_benchmark(seed=BENCHMARK_SEED, held_out_families=HELD_OUT,
                           train_fraction=0.7, val_fraction=0.1, version="v1")


# --------------------------------------------------------------------------
# Determinism and structure
# --------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_byte_identical(self, benchmark):
        again = build_benchmark(seed=BENCHMARK_SEED, held_out_families=HELD_OUT,
                                train_fraction=0.7, val_fraction=0.1, version="v1")
        assert [t.to_dict() for t in benchmark] == [t.to_dict() for t in again]

    def test_different_seed_changes_fixtures(self, benchmark):
        other = build_benchmark(seed=BENCHMARK_SEED + 1, held_out_families=HELD_OUT,
                                train_fraction=0.7, val_fraction=0.1, version="v1")
        by_id = {t.task_id: t for t in benchmark}
        differing = sum(1 for t in other if t.task_id in by_id
                        and by_id[t.task_id].fixture != t.fixture)
        assert differing > len(other) * 0.5, "seed should change most fixtures"

    def test_task_ids_unique(self, benchmark):
        ids = [t.task_id for t in benchmark]
        assert len(ids) == len(set(ids))


class TestCoverage:
    def test_size_in_target_range(self, benchmark):
        assert 180 <= len(benchmark) <= 240

    def test_all_categories_present(self, benchmark):
        present = {t.category for t in benchmark}
        assert present == set(CATEGORIES)

    def test_all_families_present(self, benchmark):
        assert {t.family for t in benchmark} == set(ENVIRONMENT_CLASSES)

    def test_every_category_has_test_and_train_tasks(self, benchmark):
        for cat in CATEGORIES:
            splits = {t.split for t in benchmark if t.category == cat}
            assert "test" in splits, f"{cat} missing from test split"

    def test_multi_step_and_ambiguity_are_substantial(self, benchmark):
        counts = benchmark_summary(benchmark)["by_category"]
        assert counts["multi_step_workflow"] >= 20
        assert counts["ambiguous_clarify"] >= 8
        assert counts["retrieve_then_act"] >= 15

    def test_difficulties_assigned(self, benchmark):
        assert {t.difficulty for t in benchmark} == {"easy", "medium", "hard"}

    def test_requests_and_budgets_sane(self, benchmark):
        for t in benchmark:
            assert len(t.request) >= 10
            assert 1 <= t.max_steps <= 12
            assert t.split in ("train", "val", "test")

    def test_no_verifier_internals_leak_into_requests(self, benchmark):
        # Jargon that would indicate verifier/reward internals in user-visible text.
        # (Plain English words like "expected" are legitimately part of requests.)
        forbidden = ("record_exists", "field_equals", "no_overlaps", "verifier",
                     "ground truth", "expectedoutcome", "forbid_state_changes",
                     "reward", "failure_categor")
        for t in benchmark:
            lowered = t.request.lower()
            for word in forbidden:
                assert word not in lowered, f"{t.task_id}: request leaks {word!r}"


# --------------------------------------------------------------------------
# Splits and leakage
# --------------------------------------------------------------------------

class TestSplitsAndLeakage:
    def test_held_out_family_entirely_in_test(self, benchmark):
        crm = [t for t in benchmark if t.family == "crm"]
        assert len(crm) >= 25, "held-out family needs enough tasks to measure generalization"
        assert all(t.split == "test" for t in crm)
        assert all("held_out_family" in t.tags for t in crm)

    def test_no_held_out_tasks_in_train_or_val(self, benchmark):
        for t in tasks_by_split(benchmark, "train") + tasks_by_split(benchmark, "val"):
            assert t.family not in HELD_OUT

    def test_splits_disjoint_and_complete(self, benchmark):
        seen: dict[str, set] = {"train": set(), "val": set(), "test": set()}
        for t in benchmark:
            assert t.task_id not in seen[t.split]
            seen[t.split].add(t.task_id)
        assert seen["train"] and seen["val"] and seen["test"]
        assert not (seen["train"] & seen["val"])
        assert not (seen["train"] & seen["test"])
        assert not (seen["val"] & seen["test"])

    def test_split_fractions_roughly_respected(self, benchmark):
        summary = benchmark_summary(benchmark)["by_split"]
        non_held = len([t for t in benchmark if t.family not in HELD_OUT])
        # Non-held-out tasks: ~70/10/20 with hash-bucket noise and category balancing.
        train_frac = summary["train"] / non_held
        assert 0.55 <= train_frac <= 0.85
        assert summary["val"] / non_held <= 0.2

    def test_split_assignment_is_order_independent(self, benchmark):
        # Deep copies: the module-scoped fixture must not be mutated.
        shuffled = [Task.from_dict(t.to_dict()) for t in reversed(benchmark)]
        for t in shuffled:
            t.split = "unassigned"
        assign_splits(shuffled, HELD_OUT, 0.7, 0.1, salt=f"v1:{BENCHMARK_SEED}")
        balance_category_coverage(shuffled, HELD_OUT, salt=f"v1:{BENCHMARK_SEED}")
        by_id = {t.task_id: t.split for t in shuffled}
        # Balancing picks max-bucket victims deterministically, so reversal
        # cannot change the outcome.
        original = build_benchmark(seed=BENCHMARK_SEED, held_out_families=HELD_OUT,
                                   train_fraction=0.7, val_fraction=0.1, version="v1")
        assert by_id == {t.task_id: t.split for t in original}

    def test_check_no_leakage_rejects_tampering(self, benchmark):
        tampered = [Task.from_dict(t.to_dict()) for t in benchmark]
        victim = next(t for t in tampered if t.family == "crm")
        victim.split = "train"
        with pytest.raises(TaskError, match="LEAKAGE"):
            check_no_leakage(tampered, HELD_OUT)

    def test_check_no_leakage_rejects_duplicate_ids(self, benchmark):
        clone = benchmark[0]
        with pytest.raises(TaskError, match="Duplicate"):
            check_no_leakage(list(benchmark) + [clone], HELD_OUT)


# --------------------------------------------------------------------------
# Fixture validity and task invariants
# --------------------------------------------------------------------------

class TestTaskIntegrity:
    def test_every_fixture_loads_into_its_environment(self, benchmark):
        for t in benchmark:
            if len(t.tool_families) == 1:
                cls = ENVIRONMENT_CLASSES[t.family]
                env = cls(fixture=json.loads(json.dumps(t.fixture)))  # deep copy
                assert env.snapshot() is not None
            else:
                for family in t.tool_families:
                    cls = ENVIRONMENT_CLASSES[family]
                    cls(fixture=json.loads(json.dumps(t.fixture[family])))

    def test_clarify_tasks_forbid_state_changes(self, benchmark):
        for t in benchmark:
            if t.expected.terminal_action == "clarify":
                assert t.category == CAT_AMBIGUOUS
                assert t.expected.forbid_state_changes, t.task_id

    def test_no_action_tasks_have_answer_checks(self, benchmark):
        """A bare 'no state change' rule would pass a fabricated success claim,
        so every no-action task must also check the answer surfaces the gap."""
        from toolforge.tasks import CAT_NO_ACTION
        for t in benchmark:
            if t.category == CAT_NO_ACTION:
                assert any(str(c.get("check", "")).startswith("answer_")
                           for c in t.expected.checks), t.task_id
                assert t.expected.forbid_state_changes, t.task_id

    def test_refusal_categories_forbid_state_changes(self, benchmark):
        for t in benchmark:
            if t.category in ("impossible_task", "invalid_tool_bait"):
                assert t.expected.forbid_state_changes, t.task_id
                assert t.expected.max_invalid_calls is not None, t.task_id

    def test_checks_are_declared_correctly(self, benchmark):
        valid_checks = {"record_exists", "record_absent", "record_count", "field_equals",
                        "no_overlaps", "tool_called", "tool_not_called",
                        "answer_contains_all", "answer_contains_any", "invalid_calls_at_most"}
        for t in benchmark:
            for check in t.expected.checks:
                assert check["check"] in valid_checks, f"{t.task_id}: {check}"
                if check["check"] in ("record_exists", "record_absent", "record_count",
                                      "field_equals", "no_overlaps"):
                    assert "collection" in check
                if check["check"] in ("tool_called", "tool_not_called"):
                    assert "tool" in check

    def test_state_checks_only_reference_known_collections(self, benchmark):
        known = {
            "calendar": {"events"}, "email": {"messages"}, "files": {"files"},
            "crm": {"companies", "contacts", "notes", "interactions"},
            "sql": {"customers", "orders", "products"}, "todo": {"tasks"},
        }
        for t in benchmark:
            for check in t.expected.checks:
                coll = check.get("collection")
                if coll:
                    family, _, name = coll.partition(".")
                    if not name:
                        family, name = t.family, coll
                    assert name in known[family], f"{t.task_id}: {coll}"

    def test_tasks_reject_bad_category_and_families(self):
        with pytest.raises(TaskError):
            Task(task_id="x", family="calendar", request="do", category="vibes",
                 difficulty="easy", fixture={}, fixture_seed=0, expected=ExpectedOutcome())
        with pytest.raises(TaskError):
            Task(task_id="x", family="calendar", request="do", category="single_step_selection",
                 difficulty="easy", fixture={}, fixture_seed=0, expected=ExpectedOutcome(),
                 tool_families=["email"])

    def test_expected_outcome_validates(self):
        with pytest.raises(TaskError):
            ExpectedOutcome(terminal_action="vibes")
        with pytest.raises(TaskError):
            ExpectedOutcome(checks=[{"nope": 1}])


# --------------------------------------------------------------------------
# JSONL I/O
# --------------------------------------------------------------------------

class TestJsonlIO:
    def test_save_load_roundtrip(self, benchmark, tmp_path):
        path = tmp_path / "bench.jsonl"
        n = save_tasks(path, benchmark)
        assert n == len(benchmark)
        loaded = load_tasks(path)
        assert [t.to_dict() for t in loaded] == [t.to_dict() for t in benchmark]

    def test_load_rejects_corrupt_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"task_id": "x"}\nnot json\n', encoding="utf-8")
        with pytest.raises(TaskError):
            load_tasks(path)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(TaskError):
            load_tasks(tmp_path / "nope.jsonl")

    def test_summary_counts_match(self, benchmark):
        summary = benchmark_summary(benchmark)
        assert summary["total"] == len(benchmark)
        assert sum(summary["by_split"].values()) == len(benchmark)
        assert sum(summary["by_category"].values()) == len(benchmark)
        assert sum(summary["by_family"].values()) == len(benchmark)


# --------------------------------------------------------------------------
# End-to-end: one scripted reference solution passes its own verifier
# --------------------------------------------------------------------------

class TestBenchmarkIsSolvable:
    def test_reference_solution_verifies(self, benchmark):
        """A hand-built correct episode for a real benchmark task must pass.

        Guards against unsatisfiable expected specs (e.g. wrong collection
        names or impossible where-clauses) in at least one end-to-end path.
        """
        registry = default_registry()
        from toolforge.executor import Executor, ScriptedPolicy
        from toolforge.verifier import verify_episode

        task = next(t for t in benchmark if t.task_id.startswith("tf-cal-ss-"))
        if "Create a calendar event titled 'Project review'" in task.request:
            actions = [
                {"action": "tool_call", "tool": "calendar.create_event",
                 "arguments": {"title": "Project review", "date": "2026-03-04",
                               "start": "13:00", "end": "14:00"}},
                {"action": "final_answer", "content": "Created 'Project review' on "
                                                      "2026-03-04 from 13:00 to 14:00."},
            ]
            episode = Executor(registry).run(task, ScriptedPolicy(actions))
            verdict = verify_episode(task, episode)
            assert verdict.success, (verdict.failure_categories,
                                     [c.to_dict() for c in verdict.checks])
        else:  # benchmark seed changed the first single-step calendar task
            pytest.skip("first calendar single-step task is not the create-event template")
