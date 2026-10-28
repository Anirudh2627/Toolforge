"""Headless tests for demo/core.py (no streamlit needed).

The Streamlit UI itself is presentation-only; these tests cover the logic it
renders: task menus (without leaking expected outcomes), freeform tasks,
policy building with fallbacks, episode driving, verdict explanations, and
JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import demo.core as core
from toolforge.tasks import Task


@pytest.fixture(scope="module")
def tasks_file(tmp_path_factory) -> str:
    """A tiny real benchmark file (built, not hand-written)."""
    from toolforge.tasks import save_tasks
    from toolforge.tasks_builders import build_benchmark
    tasks = build_benchmark(seed=20260301)
    keep = [t for t in tasks if t.task_id in
            ("tf-cal-ss-001", "tf-cal-amb-001", "tf-cal-imp-001", "tf-crm-ss-001")]
    path = tmp_path_factory.mktemp("demo") / "tasks.jsonl"
    save_tasks(path, keep)
    return str(path)


class TestTaskMenu:
    def test_menu_never_exposes_expected_outcomes(self, tasks_file):
        entries = core.list_benchmark_tasks(tasks_file)
        assert entries
        serialized = json.dumps(entries)
        for forbidden in ("expected", "checks", "record_exists", "fixture"):
            assert forbidden not in serialized

    def test_filters(self, tasks_file):
        cal = core.list_benchmark_tasks(tasks_file, family="calendar")
        assert all(e["family"] == "calendar" for e in cal)
        amb = core.list_benchmark_tasks(tasks_file, category="ambiguous_clarify")
        assert all(e["category"] == "ambiguous_clarify" for e in amb)

    def test_load_task(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        assert isinstance(task, Task) and task.family == "calendar"
        with pytest.raises(KeyError):
            core.load_benchmark_task("nope", tasks_file)


class TestFreeform:
    def test_freeform_task_shape(self):
        task = core.make_freeform_task("Do something", "calendar", seed=3)
        assert task.task_id == "freeform-calendar"
        assert task.expected.terminal_action == "any"
        assert task.fixture["events"]  # seeded default fixture

    def test_freeform_run_is_honest_about_verification(self):
        task = core.make_freeform_task("Hello?", "todo", seed=1)
        policy, desc = core.build_policy("scripted-final")
        outcome = core.run_demo(task, policy)
        assert outcome.verified_task is False
        lines = core.explain_outcome(outcome)
        assert any("no ground truth" in l.lower() or "freeform" in l.lower() for l in lines)


class TestPolicies:
    def test_reference_policy_on_benchmark_task(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        policy, desc = core.build_policy("reference", task=task)
        assert "reference" in desc
        outcome = core.run_demo(task, policy)
        assert outcome.success, outcome.failure_categories

    def test_reference_policy_clarify_task(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-amb-001", tasks_file)
        policy, _ = core.build_policy("reference", task=task)
        outcome = core.run_demo(task, policy)
        assert outcome.success and outcome.termination == "clarify"

    def test_reference_falls_back_when_not_compilable(self, tasks_file):
        """A task whose spec the compiler cannot synthesize must fall back to
        the random baseline (loudly described), never crash the demo."""
        from toolforge.tasks import ExpectedOutcome
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        task.expected = ExpectedOutcome(checks=[
            {"check": "field_equals", "collection": "events",
             "where": {"id": "evt_999"}, "field": "title", "value": "x"}])
        policy, desc = core.build_policy("reference", task=task)
        assert "random" in desc and "no compiled reference" in desc

    def test_hf_mode_requires_preloaded_model(self):
        with pytest.raises(ValueError, match="never downloads at startup"):
            core.build_policy("hf", model_id="x")

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            core.build_policy("gpt-magic")


class TestRunAndExplain:
    def test_success_explanation_and_export(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        policy, _ = core.build_policy("reference", task=task)
        outcome = core.run_demo(task, policy)
        assert outcome.success
        lines = core.explain_outcome(outcome)
        assert any("SUCCESS" in l for l in lines)
        assert any("Environment changes" in l for l in lines)
        blob = core.outcome_to_dict(outcome)
        json.dumps(blob)  # serializable
        assert blob["reward"]["success"] is True
        assert blob["steps"][0]["tool"] == "calendar.create_event"

    def test_failure_explanation_lists_categories(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        policy, _ = core.build_policy("scripted-final")  # claims done without acting
        outcome = core.run_demo(task, policy)
        assert not outcome.success
        assert "state_check_failed" in outcome.failure_categories
        lines = core.explain_outcome(outcome)
        assert any("state_check_failed" in l for l in lines)
        assert any("actual state" in l for l in lines)

    def test_impossible_task_refusal_verified(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-imp-001", tasks_file)
        policy, _ = core.build_policy("reference", task=task)
        outcome = core.run_demo(task, policy)
        assert outcome.success
        assert outcome.state_changes == []

    def test_reward_present_and_verifier_backed(self, tasks_file):
        task = core.load_benchmark_task("tf-cal-ss-001", tasks_file)
        policy, _ = core.build_policy("reference", task=task)
        outcome = core.run_demo(task, policy)
        assert outcome.reward["components"]["task_success"] == 1.0


class TestFamiliesAndDocs:
    def test_available_families(self):
        assert core.available_families() == ["calendar", "crm", "email", "files", "sql", "todo"]

    def test_grammar_note_mentions_all_actions(self):
        note = core.render_policy_context_note()
        for action in ("tool_call", "clarify", "final_answer"):
            assert action in note


class TestAppSyntax:
    def test_app_parses_without_importing_streamlit(self):
        """app.py must be syntactically valid; importing it requires streamlit,
        which the test suite deliberately does not."""
        import ast
        source = (Path(__file__).resolve().parents[1] / "demo" / "app.py").read_text()
        ast.parse(source)
