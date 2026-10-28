"""Reward function tests: component math, config dominance, adversarial cases.

The invariants that matter for RL integrity:
- verified state-change success is the ONLY path to a positive reward;
- claiming success without acting earns the failure reward;
- penalties can never be outweighed by formatting;
- invalid-call spam and loop no-ops strictly decrease reward;
- configs where shaping could dominate task success are rejected at startup.
"""

from __future__ import annotations

import pytest

from toolforge.executor import EpisodeResult, ScriptedPolicy
from toolforge.rewards import RewardConfig, RewardConfigError, compute_reward
from toolforge.verifier import VerificationResult, verify_episode

from conftest import make_calendar_task

CREATE = {"action": "tool_call", "tool": "calendar.create_event",
          "arguments": {"title": "Sync", "date": "2026-03-04", "start": "10:00", "end": "11:00"}}
FINAL_OK = {"action": "final_answer", "content": "Created the event."}
FINAL_LIE = {"action": "final_answer", "content": "I have successfully created the event!"}


def ep(termination="final_answer", tool_calls=0, invalid=0, parse_errors=0, turns=1):
    return EpisodeResult(task_id="t", termination=termination, tool_call_count=tool_calls,
                         invalid_call_count=invalid, parse_error_count=parse_errors,
                         trace=[object()] * turns)


def verdict(success, categories=()):
    return VerificationResult(task_id="t", success=success, failure_categories=list(categories))


class TestComponents:
    def test_verified_success_baseline(self):
        r = compute_reward(ep(tool_calls=1), verdict(True), RewardConfig(), max_steps=8)
        assert r.success
        assert r.total == pytest.approx(1.0 - 0.005)
        assert r.components["task_success"] == 1.0
        assert r.components["tool_calls"] == pytest.approx(-0.005)

    def test_claimed_success_without_acting_gets_failure_reward(self):
        r = compute_reward(ep(tool_calls=0), verdict(False), RewardConfig(), max_steps=8)
        assert r.total == pytest.approx(0.0)
        assert r.components["task_success"] == 0.0

    def test_success_beats_lie(self):
        cfg = RewardConfig()
        honest = compute_reward(ep(tool_calls=1), verdict(True), cfg, max_steps=8)
        lie = compute_reward(ep(tool_calls=0), verdict(False), cfg, max_steps=8)
        assert honest.total > lie.total

    def test_invalid_spam_strictly_decreases(self):
        cfg = RewardConfig()
        totals = [compute_reward(ep(invalid=n, turns=n + 1), verdict(False), cfg,
                                 max_steps=8).total for n in range(5)]
        assert totals == sorted(totals, reverse=True)
        assert totals[-1] < -0.05

    def test_step_limit_penalty(self):
        r = compute_reward(ep(termination="step_limit", tool_calls=8, turns=8),
                           verdict(False, ["step_limit_exceeded"]), RewardConfig(), max_steps=8)
        assert r.components["step_limit"] == -0.10
        assert r.total < 0

    def test_loop_guard_noop_does_not_accumulate_positive(self):
        r = compute_reward(ep(termination="loop_guard", tool_calls=4, turns=4),
                           verdict(False, ["loop_guard"]), RewardConfig(), max_steps=8)
        assert r.components["loop_guard"] == -0.10
        assert r.total < 0  # repeated identical calls can never pay

    def test_correct_clarification_is_full_success(self):
        r = compute_reward(ep(termination="clarify", tool_calls=0), verdict(True),
                           RewardConfig(), max_steps=8)
        assert r.total == pytest.approx(1.0)

    def test_unnecessary_clarification_penalized(self):
        r = compute_reward(ep(termination="clarify"), verdict(False, ["unnecessary_clarification"]),
                           RewardConfig(), max_steps=8)
        assert r.components["wrong_clarification"] == -0.10
        assert r.total == pytest.approx(-0.10)

    def test_missing_clarification_penalized(self):
        r = compute_reward(ep(), verdict(False, ["missing_clarification"]), RewardConfig(),
                           max_steps=8)
        assert r.components["wrong_clarification"] == -0.10

    def test_policy_error_penalized(self):
        r = compute_reward(ep(termination="policy_error"), verdict(False, ["policy_error"]),
                           RewardConfig(), max_steps=8)
        assert r.components["policy_error"] == -0.20

    def test_format_bonus_off_by_default_and_capped(self):
        r = compute_reward(ep(), verdict(True), RewardConfig(), max_steps=8)
        assert "format" not in r.components
        cfg = RewardConfig(format_bonus=0.05)
        clean = compute_reward(ep(parse_errors=0), verdict(True), cfg, max_steps=8)
        dirty = compute_reward(ep(parse_errors=1, invalid=1), verdict(True), cfg, max_steps=8)
        assert clean.components.get("format") == 0.05
        assert "format" not in dirty.components

    def test_all_components_logged(self):
        r = compute_reward(ep(tool_calls=2, invalid=1, parse_errors=1),
                           verdict(False, ["state_check_failed"]), RewardConfig(), max_steps=8)
        assert set(r.components) >= {"task_success", "invalid_calls", "tool_calls"}
        assert r.to_dict()["total"] == pytest.approx(r.total)


class TestConfigValidation:
    def test_dominance_violation_rejected(self):
        cfg = RewardConfig(success_reward=0.05, tool_call_penalty=-0.02,
                           invalid_call_penalty=-0.02)
        with pytest.raises(RewardConfigError, match="dominance"):
            cfg.validate(max_steps=8)

    def test_dominance_ok_for_defaults(self):
        RewardConfig().validate(max_steps=12)  # must not raise

    def test_format_bonus_capped_at_10_percent(self):
        with pytest.raises(RewardConfigError, match="format_bonus"):
            RewardConfig(format_bonus=0.2).validate(max_steps=8)

    def test_positive_penalties_rejected(self):
        with pytest.raises(RewardConfigError):
            RewardConfig(invalid_call_penalty=0.1).validate(max_steps=8)

    def test_success_must_be_positive(self):
        with pytest.raises(RewardConfigError):
            RewardConfig(success_reward=0.0).validate(max_steps=8)

    def test_failure_below_success(self):
        with pytest.raises(RewardConfigError):
            RewardConfig(success_reward=0.5, failure_reward=0.5).validate(max_steps=8)


class TestOnRealEpisodes:
    """End-to-end through executor+verifier: reward must track verified state."""

    def test_real_success_reward(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        episode = executor.run(task, ScriptedPolicy([CREATE, FINAL_OK]))
        v = verify_episode(task, episode)
        r = compute_reward(episode, v, max_steps=task.max_steps)
        assert v.success and r.total == pytest.approx(1.0 - 0.005)

    def test_real_lie_reward_zero(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        episode = executor.run(task, ScriptedPolicy([FINAL_LIE]))
        v = verify_episode(task, episode)
        r = compute_reward(episode, v, max_steps=task.max_steps)
        assert not v.success
        assert r.total == pytest.approx(0.0)  # a confident lie earns nothing
