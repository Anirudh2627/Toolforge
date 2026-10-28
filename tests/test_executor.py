"""Executor tests: episode loop, validation feedback, loop guard, step limit,
context hygiene (no expected-outcome leakage), multi-family routing, determinism.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_calendar_task
from toolforge.config import EnvSettings
from toolforge.executor import Executor, PolicyContext, ScriptedPolicy
from toolforge.tasks import ExpectedOutcome, Task

CREATE_EVENT = {"action": "tool_call", "tool": "calendar.create_event",
                "arguments": {"title": "Sync", "date": "2026-03-04",
                              "start": "10:00", "end": "11:00"}}
FINAL = {"action": "final_answer", "content": "Done."}


class RecordingPolicy:
    """Captures every PolicyContext; replays scripted outputs."""

    def __init__(self, outputs: list[str]):
        self.contexts: list[PolicyContext] = []
        self._outputs = list(outputs)

    def __call__(self, context: PolicyContext) -> str:
        self.contexts.append(context)
        if not self._outputs:
            return json.dumps(FINAL)
        return self._outputs.pop(0)


class TestHappyPath:
    def test_scripted_success_episode(self, executor, small_calendar_fixture):
        task = make_calendar_task(request="Create the Sync event.", fixture=small_calendar_fixture)
        episode = executor.run(task, ScriptedPolicy([CREATE_EVENT, FINAL]))
        assert episode.termination == "final_answer"
        assert episode.final_output == "Done."
        assert episode.tool_call_count == 1
        assert episode.invalid_call_count == 0
        titles = [e["title"] for e in episode.final_states["calendar"]["events"]]
        assert "Sync" in titles
        assert "Sync" not in [e["title"] for e in episode.initial_states["calendar"]["events"]]

    def test_trace_structure(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        episode = executor.run(task, ScriptedPolicy([CREATE_EVENT, FINAL]))
        steps = [s.to_dict() for s in episode.trace]
        assert steps[0]["tool"] == "calendar.create_event"
        assert steps[0]["tool_result"]["ok"] is True
        assert steps[0]["action"]["action"] == "tool_call"
        assert steps[1]["action"]["action"] == "final_answer"
        json.dumps(episode.to_dict())  # fully serializable

    def test_clarify_terminates_episode(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  terminal="clarify", forbid=True)
        clarify = {"action": "clarify", "question": "Which date?"}
        episode = executor.run(task, ScriptedPolicy([clarify]))
        assert episode.termination == "clarify"
        assert episode.final_output == "Which date?"
        assert episode.tool_call_count == 0


class TestErrorFeedback:
    def test_parse_error_is_reported_and_episode_continues(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        policy = RecordingPolicy(["I will create the event now!",
                                  json.dumps(CREATE_EVENT)])
        episode = executor.run(task, policy)
        assert episode.parse_error_count == 1
        assert episode.invalid_call_count == 1
        assert episode.termination == "final_answer"
        # The policy saw a structured error observation on the tool channel,
        # carrying the specific parse-failure category.
        second_ctx = policy.contexts[1]
        tool_msgs = [m for m in second_ctx.messages if m["role"] == "tool"]
        assert json.loads(tool_msgs[-1]["content"])["error"]["code"] == "invalid_json"

    def test_unknown_tool(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        bogus = {"action": "tool_call", "tool": "calendar.teleport", "arguments": {}}
        episode = executor.run(task, ScriptedPolicy([bogus, FINAL]))
        assert episode.invalid_call_count == 1
        assert episode.tool_call_count == 0
        step = episode.trace[0].to_dict()
        assert step["tool_result"]["error"]["code"] == "unknown_tool"

    def test_schema_violation(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        missing_args = {"action": "tool_call", "tool": "calendar.create_event",
                        "arguments": {"title": "No date or times"}}
        episode = executor.run(task, ScriptedPolicy([missing_args, FINAL]))
        assert episode.invalid_call_count == 1
        issues = episode.trace[0].validation_issues
        assert issues and {i["kind"] for i in issues} == {"missing_required"}

    def test_tool_level_failure_is_observable(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        bad_time = {"action": "tool_call", "tool": "calendar.create_event",
                    "arguments": {"title": "X", "date": "2026-03-04",
                                  "start": "12:00", "end": "11:00"}}
        episode = executor.run(task, ScriptedPolicy([bad_time, FINAL]))
        # A tool-level error is an executed call, not an invalid call.
        assert episode.tool_call_count == 1
        assert episode.invalid_call_count == 0
        result = episode.trace[0].tool_result
        assert result["ok"] is False and result["error"]["code"] == "invalid_argument"


class TestTerminationGuards:
    def test_step_limit(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=3)

        def endless(ctx: PolicyContext) -> str:
            # Vary arguments so the loop guard does not trigger first.
            return json.dumps({"action": "tool_call", "tool": "calendar.list_events",
                               "arguments": {"date": f"2026-03-{ctx.step + 10:02d}"}})

        episode = executor.run(task, endless)
        assert episode.termination == "step_limit"
        assert len(episode.trace) == 3
        assert any("step_limit" in w for w in episode.warnings)

    def test_loop_guard(self, registry, small_calendar_fixture):
        executor = Executor(registry, settings=EnvSettings(default_max_steps=20, repeat_limit=2))
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=20)
        same_call = {"action": "tool_call", "tool": "calendar.list_events", "arguments": {}}
        episode = executor.run(task, ScriptedPolicy([same_call] * 10))
        assert episode.termination == "loop_guard"
        # repeat_limit=2: the third identical call triggers termination.
        assert episode.tool_call_count == 3
        assert any("loop_guard" in w for w in episode.warnings)

    def test_policy_must_return_str(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        with pytest.raises(TypeError):
            executor.run(task, lambda ctx: {"action": "final_answer", "content": "x"})


class TestContextHygiene:
    def test_expected_outcome_never_visible_to_policy(self, executor, small_calendar_fixture):
        secret_value = "SECRET-GROUND-TRUTH-42"
        task = make_calendar_task(
            request="Create the Sync event.",
            fixture=small_calendar_fixture,
            checks=[{"check": "record_exists", "collection": "events",
                     "where": {"title": secret_value}}],
        )
        policy = RecordingPolicy([json.dumps(CREATE_EVENT)])
        executor.run(task, policy)
        for ctx in policy.contexts:
            visible = json.dumps(ctx.to_dict())
            assert secret_value not in visible
            assert "record_exists" not in visible
            assert "expected" not in visible.lower()

    def test_system_prompt_contains_grammar_and_tools(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        policy = RecordingPolicy([json.dumps(FINAL)])
        executor.run(task, policy)
        system = policy.contexts[0].messages[0]
        assert system["role"] == "system"
        assert "tool_call" in system["content"] and "final_answer" in system["content"]
        assert "calendar.create_event" in system["content"]
        assert "2026-03-02" in system["content"]  # simulated current date


class TestMultiFamily:
    def test_routes_tools_across_families(self, executor, small_calendar_fixture):
        email_fixture = {
            "current_date": "2026-03-02", "user_email": "user@example.com",
            "folders": ["inbox", "archive", "spam", "sent", "drafts"],
            "next_message_num": 1, "messages": [],
        }
        task = Task(
            task_id="test-multi-001", family="calendar",
            request="Create the event and email alice.",
            category="multi_step_workflow", difficulty="medium",
            fixture={"calendar": small_calendar_fixture, "email": email_fixture},
            fixture_seed=0,
            expected=ExpectedOutcome(terminal_action="final_answer", checks=[]),
            tool_families=["calendar", "email"], max_steps=8,
            source="handwritten", benchmark_seed=0, benchmark_version="test",
        )
        actions = [
            CREATE_EVENT,
            {"action": "tool_call", "tool": "email.send_email",
             "arguments": {"to": "alice@example.com", "subject": "Sync", "body": "Created."}},
            FINAL,
        ]
        episode = executor.run(task, ScriptedPolicy(actions))
        assert episode.termination == "final_answer"
        assert episode.tool_call_count == 2
        assert "calendar" in episode.final_states and "email" in episode.final_states
        assert any(m["folder"] == "sent" for m in episode.final_states["email"]["messages"])

    def test_multi_family_fixture_must_cover_all_families(self, executor, small_calendar_fixture):
        task = Task(
            task_id="test-multi-002", family="calendar", request="x",
            category="multi_step_workflow", difficulty="medium",
            fixture={"calendar": small_calendar_fixture},  # email missing
            fixture_seed=0, expected=ExpectedOutcome(),
            tool_families=["calendar", "email"], max_steps=4,
            source="handwritten", benchmark_seed=0, benchmark_version="test",
        )
        with pytest.raises(ValueError, match="missing 'email'"):
            executor.run(task, ScriptedPolicy([FINAL]))


class TestDeterminism:
    def test_same_script_same_episode(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        e1 = executor.run(task, ScriptedPolicy([CREATE_EVENT, FINAL])).to_dict()
        e2 = executor.run(task, ScriptedPolicy([CREATE_EVENT, FINAL])).to_dict()
        for e in (e1, e2):  # timings are machine-dependent by design
            e.pop("policy_seconds"), e.pop("env_seconds")
            for step in e["trace"]:
                step.pop("policy_seconds"), step.pop("env_seconds")
        assert e1 == e2

    def test_environment_isolation_between_episodes(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        executor.run(task, ScriptedPolicy([CREATE_EVENT, FINAL]))
        episode2 = executor.run(task, ScriptedPolicy([FINAL]))
        titles = [e["title"] for e in episode2.final_states["calendar"]["events"]]
        assert "Sync" not in titles  # episode 1's write must not leak into episode 2

    def test_scripted_policy_exhaustion_is_flagged(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=5)
        episode = executor.run(task, ScriptedPolicy([]))
        assert episode.termination == "final_answer"
        assert any("script_exhausted" in w for w in episode.warnings)
