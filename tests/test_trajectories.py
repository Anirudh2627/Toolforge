"""Trajectory record tests: fidelity of serialization, split guards, round-trips."""

from __future__ import annotations

import json

import pytest

from conftest import make_calendar_task
from toolforge.executor import ScriptedPolicy
from toolforge.trajectories import (
    Trajectory,
    TrajectoryError,
    load_trajectories,
    messages_from_episode,
    save_trajectories,
    trajectory_from_episode,
    trajectory_id,
)
from toolforge.verifier import verify_episode

CREATE = {"action": "tool_call", "tool": "calendar.create_event",
          "arguments": {"title": "Sync", "date": "2026-03-04", "start": "10:00", "end": "11:00"}}
FINAL = {"action": "final_answer", "content": "Created."}


@pytest.fixture
def success_episode(executor, small_calendar_fixture):
    task = make_calendar_task(fixture=small_calendar_fixture,
                              checks=[{"check": "record_exists", "collection": "events",
                                       "where": {"title": "Sync"}}])
    episode = executor.run(task, ScriptedPolicy([CREATE, FINAL]))
    episode.system_prompt = "SYS"
    return task, episode, verify_episode(task, episode)


class TestTrajectoryFromEpisode:
    def test_fields_and_message_fidelity(self, success_episode):
        task, episode, verdict = success_episode
        traj = trajectory_from_episode(task, episode, verdict, source="reference",
                                       policy_id="test", tools_fingerprint="fp123")
        assert traj.verified is True
        assert traj.source == "reference"
        assert traj.tools_fingerprint == "fp123"
        msgs = traj.messages
        assert msgs[0] == {"role": "system", "content": episode.system_prompt}
        assert msgs[1] == {"role": "user", "content": task.request}
        assert msgs[2]["role"] == "assistant"
        assert json.loads(msgs[2]["content"])["tool"] == "calendar.create_event"
        assert msgs[3]["role"] == "tool"
        assert json.loads(msgs[3]["content"])["ok"] is True
        assert msgs[-1]["role"] == "assistant"
        assert traj.actions == [s.action for s in episode.trace if s.action]
        assert traj.tool_call_count == 1

    def test_roundtrip(self, success_episode):
        task, episode, verdict = success_episode
        traj = trajectory_from_episode(task, episode, verdict, source="self", policy_id="p")
        restored = Trajectory.from_dict(json.loads(json.dumps(traj.to_dict())))
        assert restored == traj

    def test_test_split_refused(self, success_episode):
        task, episode, verdict = success_episode
        task.split = "test"
        with pytest.raises(TrajectoryError, match="test-split"):
            trajectory_from_episode(task, episode, verdict, source="reference", policy_id="p")

    def test_unknown_source_refused(self, success_episode):
        task, episode, verdict = success_episode
        with pytest.raises(TrajectoryError, match="source"):
            trajectory_from_episode(task, episode, verdict, source="gpt-magic", policy_id="p")

    def test_from_dict_rejects_unknown_fields(self, success_episode):
        task, episode, verdict = success_episode
        traj = trajectory_from_episode(task, episode, verdict, source="reference", policy_id="p")
        data = traj.to_dict()
        data["hallucinated_field"] = 1
        with pytest.raises(TrajectoryError, match="Unknown trajectory field"):
            Trajectory.from_dict(data)


class TestPersistence:
    def test_save_load(self, success_episode, tmp_path):
        task, episode, verdict = success_episode
        trajs = [trajectory_from_episode(task, episode, verdict, source="reference",
                                         policy_id="p", index=i) for i in range(2)]
        path = tmp_path / "trajs.jsonl"
        save_trajectories(path, trajs)
        loaded = load_trajectories(path)
        assert [t.to_dict() for t in loaded] == [t.to_dict() for t in trajs]

    def test_load_missing(self, tmp_path):
        with pytest.raises(TrajectoryError):
            load_trajectories(tmp_path / "nope.jsonl")

    def test_id_deterministic_and_unique(self):
        a = trajectory_id("tf-cal-ss-001", "reference", "run1", 0)
        b = trajectory_id("tf-cal-ss-001", "reference", "run1", 0)
        c = trajectory_id("tf-cal-ss-001", "reference", "run1", 1)
        assert a == b and a != c and a.startswith("traj-tf-cal-ss-001-reference-")


class TestMessagesFromEpisode:
    def test_parse_failure_turns_included(self, executor, small_calendar_fixture):
        """Trajectories faithfully record invalid turns (SFT filtering drops them later)."""
        task = make_calendar_task(fixture=small_calendar_fixture)
        episode = executor.run(task, ScriptedPolicy(["oops not json", CREATE, FINAL]))
        msgs = messages_from_episode(task, episode)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert json.loads(msgs[3]["content"])["ok"] is False
