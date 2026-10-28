"""Verifier tests: state-based success, check semantics, failure categories,
volatile-key handling, and loud failure on malformed specs.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_calendar_task
from toolforge.executor import EpisodeResult, ScriptedPolicy, TraceStep
from toolforge.tasks import ExpectedOutcome, Task
from toolforge.verifier import (
    VerificationResult,
    VerifierError,
    diff_states,
    evaluate_check,
    match_value,
    match_where,
    verify_episode,
)

CREATE_EVENT = {"action": "tool_call", "tool": "calendar.create_event",
                "arguments": {"title": "Sync", "date": "2026-03-04",
                              "start": "10:00", "end": "11:00"}}
FINAL = {"action": "final_answer", "content": "Created the Sync event for 2026-03-04."}


def run(executor, task, actions) -> EpisodeResult:
    return executor.run(task, ScriptedPolicy(actions))


# --------------------------------------------------------------------------
# Matching primitives
# --------------------------------------------------------------------------

class TestMatching:
    def test_scalar_equality(self):
        assert match_value("a", "a") and not match_value("a", "b")

    def test_list_membership_and_set_equality(self):
        assert match_value(["x", "y"], "x")            # scalar vs list -> membership
        assert match_value(["x", "y"], ["y", "x"])     # list vs list -> order-insensitive
        assert not match_value(["x"], ["x", "y"])

    def test_ops(self):
        assert match_value("hello world", {"op": "contains", "value": "World"})  # case-insens.
        assert not match_value("hello", {"op": "contains", "value": "bye"})
        assert match_value(5, {"op": "gt", "value": 4})
        assert match_value(5, {"op": "gte", "value": 5})
        assert match_value("a", {"op": "in", "value": ["a", "b"]})
        assert match_value("c", {"op": "not_in", "value": ["a", "b"]})
        assert match_value(None, {"op": "exists", "value": False})
        assert match_value("x", {"op": "ne", "value": "y"})
        with pytest.raises(VerifierError):
            match_value("x", {"op": "teleport", "value": 1})

    def test_match_where(self):
        record = {"id": "e1", "tags": ["a", "b"], "n": 3}
        assert match_where(record, {"id": "e1", "tags": "a"})
        assert match_where(record, {"n": {"op": "gte", "value": 3}})
        assert not match_where(record, {"n": {"op": "gt", "value": 3}})


# --------------------------------------------------------------------------
# State diffing
# --------------------------------------------------------------------------

class TestDiffStates:
    def test_id_aligned_lists(self):
        initial = {"events": [{"id": "e1", "status": "scheduled", "title": "A"}]}
        final = {"events": [{"id": "e1", "status": "cancelled", "title": "A"}]}
        changes = diff_states(initial, final)
        assert changes == ["events[id=e1].status: 'scheduled' -> 'cancelled'"]

    def test_added_removed_records(self):
        initial = {"events": [{"id": "e1"}]}
        final = {"events": [{"id": "e2"}]}
        changes = diff_states(initial, final)
        assert any("e1" in c and "removed" in c for c in changes)
        assert any("e2" in c and "added" in c for c in changes)

    def test_ignore_keys(self):
        initial = {"messages": [{"id": "m1", "read": False}]}
        final = {"messages": [{"id": "m1", "read": True}]}
        assert diff_states(initial, final) != []
        assert diff_states(initial, final, ignore_keys=frozenset({"read"})) == []

    def test_scalar_and_nested(self):
        assert diff_states({"a": 1}, {"a": 2}) == ["a: 1 -> 2"]
        assert diff_states({"a": {"b": [1]}}, {"a": {"b": [1, 2]}}) == \
               ["a.b.length: 1 -> 2"]


# --------------------------------------------------------------------------
# End-to-end verification through the executor
# --------------------------------------------------------------------------

class TestVerifyEpisode:
    def test_success_from_state_not_text(self, executor, small_calendar_fixture):
        task = make_calendar_task(
            request="Create the Sync event.", fixture=small_calendar_fixture,
            checks=[{"check": "record_exists", "collection": "events",
                     "where": {"title": "Sync", "date": "2026-03-04",
                               "start": "10:00", "end": "11:00"}}])
        verdict = verify_episode(task, run(executor, task, [CREATE_EVENT, FINAL]))
        assert verdict.success, verdict.failure_categories
        assert verdict.stats["tool_call_count"] == 1

    def test_claiming_success_without_acting_fails(self, executor, small_calendar_fixture):
        task = make_calendar_task(
            request="Create the Sync event.", fixture=small_calendar_fixture,
            checks=[{"check": "record_exists", "collection": "events",
                     "where": {"title": "Sync"}}])
        episode = run(executor, task, [
            {"action": "final_answer", "content": "I have created the Sync event successfully!"}])
        verdict = verify_episode(task, episode)
        assert not verdict.success
        assert "state_check_failed" in verdict.failure_categories

    def test_forbidden_state_change_detected(self, executor, small_calendar_fixture):
        task = make_calendar_task(
            request="How many events on 2026-03-03?", fixture=small_calendar_fixture,
            terminal="final_answer", forbid=True,
            checks=[{"check": "answer_contains_all", "strings": ["2"]}])
        # Correct answer, but the model also sneaks in a state change.
        episode = run(executor, task, [
            CREATE_EVENT,
            {"action": "final_answer", "content": "You have 2 events on 2026-03-03."}])
        verdict = verify_episode(task, episode)
        assert not verdict.success
        assert "forbidden_state_change" in verdict.failure_categories
        assert any("calendar.events" in c for c in verdict.forbidden_changes)

    def test_volatile_read_flags_do_not_count_as_changes(self, executor):
        fixture = {
            "current_date": "2026-03-02", "user_email": "user@example.com",
            "folders": ["inbox", "archive", "spam", "sent", "drafts"], "next_message_num": 2,
            "messages": [{"id": "msg_001", "from": "alice@example.com", "to": "user@example.com",
                          "subject": "Hi", "body": "Hello", "date": "2026-03-01T09:00",
                          "folder": "inbox", "read": False, "in_reply_to": None}],
        }
        task = Task(
            task_id="test-email-volatile", family="email", request="Read msg_001.",
            category="retrieve_then_act", difficulty="easy", fixture=fixture, fixture_seed=0,
            expected=ExpectedOutcome(terminal_action="final_answer", forbid_state_changes=True,
                                     checks=[{"check": "tool_called",
                                              "tool": "email.read_message"}]),
            max_steps=4, source="handwritten", benchmark_seed=0, benchmark_version="test")
        episode = run(executor, task, [
            {"action": "tool_call", "tool": "email.read_message",
             "arguments": {"message_id": "msg_001"}},
            {"action": "final_answer", "content": "It says hello."}])
        verdict = verify_episode(task, episode)
        assert verdict.success, verdict.failure_categories

    def test_missing_and_unnecessary_clarification(self, executor, small_calendar_fixture):
        clarify = {"action": "clarify", "question": "When?"}
        # Task requires clarification; model answers instead.
        task = make_calendar_task(fixture=small_calendar_fixture, terminal="clarify", forbid=True)
        verdict = verify_episode(task, run(executor, task, [FINAL]))
        assert not verdict.success and "missing_clarification" in verdict.failure_categories
        # Task requires action; model clarifies.
        task2 = make_calendar_task(fixture=small_calendar_fixture, terminal="final_answer",
                                   checks=[{"check": "record_exists", "collection": "events",
                                            "where": {"title": "Sync"}}])
        verdict2 = verify_episode(task2, run(executor, task2, [clarify]))
        assert not verdict2.success and "unnecessary_clarification" in verdict2.failure_categories

    def test_step_limit_violation_fails_even_with_correct_state(self, executor,
                                                                 small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=1,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        # Model creates the event but never emits a terminal action within budget.
        verdict = verify_episode(task, run(executor, task, [CREATE_EVENT]))
        assert not verdict.success
        assert "step_limit_exceeded" in verdict.failure_categories

    def test_tool_called_and_not_called(self, executor, small_calendar_fixture):
        task = make_calendar_task(
            fixture=small_calendar_fixture,
            checks=[
                {"check": "tool_called", "tool": "calendar.create_event", "min_times": 1,
                 "arguments_match": {"title": "Sync"}},
                {"check": "tool_called", "tool": "calendar.create_event",
                 "arguments_match": {"title": "Wrong"}},
                {"check": "tool_not_called", "tool": "calendar.cancel_event"},
            ])
        verdict = verify_episode(task, run(executor, task, [CREATE_EVENT, FINAL]))
        assert not verdict.success
        cats = verdict.failure_categories
        assert "required_tool_not_called" in cats
        assert "forbidden_tool_called" not in cats
        passed = {c.params["arguments_match"]["title"]: c.passed
                  for c in verdict.checks
                  if c.check == "tool_called" and "arguments_match" in c.params}
        assert passed == {"Sync": True, "Wrong": False}

    def test_invalid_call_budget(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=6)
        task.expected = ExpectedOutcome(terminal_action="final_answer",
                                        forbid_state_changes=False, checks=[],
                                        max_invalid_calls=1)
        episode = run(executor, task, ["garbage", "more garbage", FINAL])
        verdict = verify_episode(task, episode)
        assert not verdict.success and "too_many_invalid_calls" in verdict.failure_categories

    def test_no_overlaps_scoped_to_subject(self, executor):
        # Fixture already contains an overlap that is NOT the model's doing.
        fixture = {
            "current_date": "2026-03-02", "user_email": "user@example.com", "next_event_num": 4,
            "events": [
                {"id": "evt_001", "title": "A", "date": "2026-03-04", "start": "10:00",
                 "end": "11:00", "attendees": [], "location": "", "notes": "",
                 "status": "scheduled"},
                {"id": "evt_002", "title": "B", "date": "2026-03-04", "start": "10:30",
                 "end": "11:30", "attendees": [], "location": "", "notes": "",
                 "status": "scheduled"},
            ],
        }
        task = make_calendar_task(
            fixture=fixture,
            checks=[{"check": "no_overlaps", "collection": "events",
                     "subject_where": {"title": "C"}, "filter": {"status": "scheduled"}}])
        good = {"action": "tool_call", "tool": "calendar.create_event",
                "arguments": {"title": "C", "date": "2026-03-04", "start": "13:00", "end": "14:00"}}
        assert verify_episode(task, run(executor, task, [good, FINAL])).success
        bad = {"action": "tool_call", "tool": "calendar.create_event",
               "arguments": {"title": "C", "date": "2026-03-04", "start": "10:45", "end": "11:15"}}
        verdict = verify_episode(task, run(executor, task, [bad, FINAL]))
        assert not verdict.success and "state_check_failed" in verdict.failure_categories


# --------------------------------------------------------------------------
# Individual check evaluation (synthetic episodes)
# --------------------------------------------------------------------------

def synthetic_episode(final_states, *, termination="final_answer", final_output="answer",
                      calls=(), invalid=0, trace_tools=()) -> EpisodeResult:
    trace = []
    for i, (tool, args, ok) in enumerate(trace_tools):
        trace.append(TraceStep(step=i, raw_output="{}",
                               action={"action": "tool_call", "tool": tool, "arguments": args},
                               tool=tool, arguments=args,
                               tool_result={"ok": ok, "output": None} if ok
                               else {"ok": False, "error": {"code": "not_found", "message": "x"}}))
    return EpisodeResult(
        task_id="synthetic", termination=termination, final_output=final_output,
        trace=trace, initial_states={}, final_states=final_states,
        tool_call_count=len(calls), invalid_call_count=invalid)


class TestEvaluateCheck:
    STATES = {"cal": {"items": [
        {"id": "a", "n": 1, "tags": ["x"]},
        {"id": "b", "n": 2, "tags": []},
    ]}}

    def ev(self, check, **params):
        return evaluate_check({"check": check, **params},
                              synthetic_episode(self.STATES), primary_family="cal")

    def test_record_count_ops(self):
        assert self.ev("record_count", collection="items", where={}, op="==", value=2).passed
        assert self.ev("record_count", collection="items", where={"n": 1}, op=">=", value=1).passed
        assert not self.ev("record_count", collection="items", where={"n": 5}, op="==", value=1).passed
        with pytest.raises(VerifierError):
            self.ev("record_count", collection="items", op="~=", value=1)

    def test_field_equals_requires_unique_match(self):
        assert self.ev("field_equals", collection="items", where={"id": "a"},
                       field="n", value=1).passed
        result = self.ev("field_equals", collection="items", where={}, field="n", value=1)
        assert not result.passed and "exactly 1" in result.detail

    def test_answer_checks(self):
        ep = synthetic_episode(self.STATES, final_output="The total is 4200 USD.")
        assert evaluate_check({"check": "answer_contains_all", "strings": ["4200", "usd"]},
                              ep, "cal").passed
        assert not evaluate_check({"check": "answer_contains_all", "strings": ["4200", "eur"]},
                                  ep, "cal").passed
        assert evaluate_check({"check": "answer_contains_any", "strings": ["eur", "4200"]},
                              ep, "cal").passed

    def test_unknown_check_raises(self):
        with pytest.raises(VerifierError):
            self.ev("vibes_check", collection="items")

    def test_unknown_collection_raises(self):
        with pytest.raises(VerifierError):
            self.ev("record_exists", collection="nonexistent")
        with pytest.raises(VerifierError):
            evaluate_check({"check": "record_exists", "collection": "ghost.items"},
                           synthetic_episode(self.STATES), "cal")

    def test_invalid_call_budget_check(self):
        ep = synthetic_episode(self.STATES, invalid=4)
        assert not evaluate_check({"check": "invalid_calls_at_most", "n": 3}, ep, "cal").passed
        assert evaluate_check({"check": "invalid_calls_at_most", "n": 4}, ep, "cal").passed


# --------------------------------------------------------------------------
# Loud-failure guarantees
# --------------------------------------------------------------------------

class TestVerifierStrictness:
    def test_task_without_expected_raises(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        task.expected = None  # simulate a corrupt task record
        with pytest.raises(VerifierError):
            verify_episode(task, run(executor, task, [FINAL]))

    def test_task_episode_mismatch_raises(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        other = make_calendar_task(task_id="other-task", fixture=small_calendar_fixture)
        with pytest.raises(VerifierError, match="mismatch"):
            verify_episode(other, run(executor, task, [FINAL]))

    def test_unknown_terminal_action_raises(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        object.__setattr__(task.expected, "terminal_action", "vibes")
        with pytest.raises(VerifierError):
            verify_episode(task, run(executor, task, [FINAL]))

    def test_verdict_is_json_serializable(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        verdict = verify_episode(task, run(executor, task, [CREATE_EVENT, FINAL]))
        assert isinstance(verdict, VerificationResult)
        json.dumps(verdict.to_dict())
