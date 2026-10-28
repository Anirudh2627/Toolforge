"""Data-generation tests: placeholder resolution, reference coverage on the real
benchmark, rejection sampling mechanics, and SFT-mix hygiene guards.
"""

from __future__ import annotations

import json

import pytest

from toolforge.data_generation import (
    CollectionStats,
    DataGenerationError,
    SolutionPolicy,
    build_sft_mix,
    collect_rollouts,
    generate_reference_trajectories,
    resolve_prev,
    write_collection,
)
from toolforge.executor import Executor, PolicyContext
from toolforge.inference import RandomPolicy
from toolforge.registry import default_registry
from toolforge.tasks import Task
from toolforge.tasks_builders import build_benchmark
from toolforge.trajectories import Trajectory


@pytest.fixture(scope="module")
def benchmark() -> list[Task]:
    return build_benchmark(seed=20260301)


@pytest.fixture(scope="module")
def train_pool(benchmark) -> list[Task]:
    return [t for t in benchmark if t.split in ("train", "val")]


class TestResolvePrev:
    def test_resolution_newest_first(self):
        outputs = [{"draft": {"id": "msg_010"}}, {"draft": {"id": "msg_013"}}]
        assert resolve_prev({"$prev": "draft.id"}, outputs) == "msg_013"

    def test_nested_structures(self):
        outputs = [{"event": {"id": "evt_005", "conflicts": []}}]
        resolved = resolve_prev({"a": {"b": [{"$prev": "event.id"}]}}, outputs)
        assert resolved == {"a": {"b": ["evt_005"]}}

    def test_missing_path_raises(self):
        with pytest.raises(KeyError):
            resolve_prev({"$prev": "ghost.id"}, [{"draft": {"id": "x"}}])

    def test_falls_back_to_older_outputs(self):
        outputs = [{"event": {"id": "evt_1"}}, {"draft": {"id": "msg_1"}}]
        assert resolve_prev({"$prev": "event.id"}, outputs) == "evt_1"


class TestSolutionPolicy:
    def test_resolves_from_conversation(self):
        policy = SolutionPolicy([
            {"action": "tool_call", "tool": "email.send_draft",
             "arguments": {"draft_id": {"$prev": "draft.id"}}},
        ])
        ctx = PolicyContext(
            request="r", tools=[], step=0, remaining_steps=4,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "{}"},
                      {"role": "tool", "content": json.dumps(
                          {"ok": True, "output": {"draft": {"id": "msg_042"}}})}])
        emitted = json.loads(policy(ctx))
        assert emitted["arguments"] == {"draft_id": "msg_042"}

    def test_exhaustion_returns_final_answer(self):
        policy = SolutionPolicy([])
        ctx = PolicyContext(request="r", tools=[], step=0, remaining_steps=2, messages=[])
        assert json.loads(policy(ctx))["action"] == "final_answer"


class TestReferenceGeneration:
    def test_full_train_pool_coverage(self, train_pool):
        """Every train/val task must yield a verified reference trajectory.

        (Guards against benchmark/compiler drift: expected specs and compiled
        solutions are derived from the same declarative checks.)"""
        executor = Executor(default_registry())
        trajs, stats = generate_reference_trajectories(train_pool, executor)
        assert stats.rejected == 0, stats.failure_categories
        coverage = len({t.task_id for t in trajs}) / len(train_pool)
        assert coverage >= 0.95, f"reference coverage dropped to {coverage:.1%}"
        assert all(t.verified for t in trajs)
        assert all(t.split in ("train", "val") for t in trajs)
        assert all(t.reward and t.reward["success"] for t in trajs)

    def test_test_split_refused(self, benchmark):
        test_task = next(t for t in benchmark if t.split == "test")
        with pytest.raises(DataGenerationError, match="test-split"):
            generate_reference_trajectories([test_task], Executor(default_registry()))

    def test_trajectories_end_with_terminal_action(self, train_pool):
        executor = Executor(default_registry())
        trajs, _ = generate_reference_trajectories(train_pool[:25], executor)
        for traj in trajs:
            last = json.loads(traj.messages[-1]["content"])
            assert last["action"] in ("final_answer", "clarify")


class TestRejectionSampling:
    def test_only_verified_kept_and_stats_recorded(self, train_pool):
        tasks = [t for t in train_pool if t.family == "calendar"][:6]
        executor = Executor(default_registry())
        trajs, stats = collect_rollouts(
            tasks, executor, lambda seed: RandomPolicy(seed=seed, max_tool_calls=2),
            samples_per_task=2, source="self", base_seed=5)
        assert stats.attempted == 12
        assert stats.kept == len(trajs)
        assert all(t.verified for t in trajs)
        assert stats.rejected == 12 - len(trajs)
        if stats.rejected:
            assert stats.failure_categories  # rejection reasons are recorded

    def test_max_kept_per_task_cap(self, train_pool):
        tasks = [t for t in train_pool if t.category == "ambiguous_clarify"][:3]
        executor = Executor(default_registry())
        # clarify tasks: an always-clarify policy passes the verifier every time
        trajs, stats = collect_rollouts(
            tasks, executor,
            lambda seed: _AlwaysClarify(seed), samples_per_task=4, source="self",
            max_kept_per_task=2)
        assert stats.verified == 12                     # every rollout passed
        assert stats.rejected == 0
        assert stats.dropped_by_cap == 6                # cap discarded the surplus
        assert stats.kept == len(trajs) == 6            # 2 per task
        assert all(t.verified for t in trajs)

    def test_deterministic_given_seed(self, train_pool):
        tasks = train_pool[:3]
        executor = Executor(default_registry())
        a, sa = collect_rollouts(tasks, executor, lambda s: RandomPolicy(seed=s),
                                 samples_per_task=2, base_seed=9)
        b, sb = collect_rollouts(tasks, executor, lambda s: RandomPolicy(seed=s),
                                 samples_per_task=2, base_seed=9)
        assert [t.trajectory_id for t in a] == [t.trajectory_id for t in b]
        assert sa.kept == sb.kept

    def test_invalid_source_label(self, train_pool):
        with pytest.raises(DataGenerationError):
            collect_rollouts(train_pool[:1], Executor(default_registry()),
                             lambda s: RandomPolicy(seed=s), source="reference")


class _AlwaysClarify:
    policy_id = "always-clarify"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, context) -> str:
        return json.dumps({"action": "clarify", "question": "What exactly should I do?"})


class TestSftMix:
    def _traj(self, task_id: str, split: str = "train", family: str = "calendar",
              verified: bool = True, source: str = "reference", clean: bool = True) -> Trajectory:
        return Trajectory(
            trajectory_id=f"traj-{task_id}-{source}", task_id=task_id, split=split,
            family=family, category="single_step_selection", difficulty="easy",
            source=source, policy_id="p", sampling_seed=0, request="r",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                      {"role": "assistant", "content": "a"}],
            actions=[], termination="final_answer", final_output="a", verified=verified,
            verification={}, parse_error_count=0 if clean else 1,
            invalid_call_count=0 if clean else 2)

    def test_filters_and_priority(self):
        ref = [self._traj("t1"), self._traj("t2", verified=False)]
        roll = [self._traj("t1", source="self"), self._traj("t3", source="self", clean=False)]
        pool, stats = build_sft_mix(ref, roll, max_per_task=1)
        # t2 dropped (unverified), t3 dropped (dirty), t1-self dropped (cap,
        # reference data has priority) -> exactly one record remains.
        assert [t.trajectory_id for t in pool] == ["traj-t1-reference"]
        assert stats["dropped"]["unverified"] == 1
        assert stats["dropped"]["not_clean"] == 1
        assert stats["dropped"]["per_task_cap"] == 1

    def test_test_split_raises(self):
        t = self._traj("t1", split="val")
        t.split = "test"  # simulate upstream corruption; mix must refuse loudly
        with pytest.raises(DataGenerationError, match="never enter SFT"):
            build_sft_mix([t])

    def test_held_out_family_raises(self):
        with pytest.raises(DataGenerationError, match="never enter SFT"):
            build_sft_mix([self._traj("t1", family="crm")])

    def test_write_collection(self, tmp_path):
        trajs = [self._traj("t1")]
        stats = CollectionStats(attempted=1, kept=1)
        path = write_collection(tmp_path, "unit", trajs, stats)
        assert path.exists()
        assert (tmp_path / "unit_stats.json").exists()
        assert json.loads((tmp_path / "unit_stats.json").read_text())["kept"] == 1


class TestCatalogCompaction:
    def test_compact_catalog_preserves_information(self, registry):
        from toolforge.data_generation import compact_catalog
        from toolforge.schemas import render_tool_catalog
        specs = registry.tool_specs(["calendar"])
        tools = [s.to_dict() for s in specs]
        compact = compact_catalog(tools)
        full = render_tool_catalog(specs)
        assert len(compact) < len(full) * 0.6, "compaction should save substantial tokens"
        for spec in specs:
            assert spec.name in compact
            assert spec.description in compact
        # required vs optional markers and type/pattern info survive
        assert "title:string" in compact and "date?:string" in compact
        assert r"\d{4}-\d{2}-\d{2}" in compact
        assert '"scheduled"|' in compact or "scheduled" in compact

    def test_compact_system_prompt_roundtrip(self, executor, small_calendar_fixture):
        from toolforge.data_generation import DataGenerationError, compact_system_prompt
        task = make_task_with_fixture(small_calendar_fixture)
        router, ctx = executor.initial_context(task)
        original = ctx.messages[0]["content"]
        compacted = compact_system_prompt(original, ctx.tools)
        assert "Available tools (name, description, JSON-schema parameters):" in compacted
        assert "- calendar.create_event(" in compacted
        assert '"parameters"' not in compacted          # JSON catalog is gone
        assert "EXACTLY ONE JSON object" in compacted   # grammar section untouched
        assert len(compacted) < len(original)
        with pytest.raises(DataGenerationError):
            compact_system_prompt("no header here", ctx.tools)

    def test_compact_trajectory_messages(self, train_pool):
        from toolforge.data_generation import (compact_trajectory_messages,
                                               generate_reference_trajectories)
        executor = Executor(default_registry())
        trajs, _ = generate_reference_trajectories(train_pool[:3], executor)
        assert trajs, "reference generation must produce trajectories"
        compacted = compact_trajectory_messages(trajs[0])
        assert compacted.messages[0]["content"] != trajs[0].messages[0]["content"]
        assert compacted.messages[1:] == trajs[0].messages[1:]
        assert compacted.verified == trajs[0].verified
        assert compacted.tools == trajs[0].tools


def make_task_with_fixture(fixture):
    from toolforge.tasks import ExpectedOutcome, Task
    return Task(task_id="t-compact", family="calendar", request="Create an event.",
                category="single_step_selection", difficulty="easy", fixture=fixture,
                fixture_seed=0, expected=ExpectedOutcome(), max_steps=4,
                source="handwritten", benchmark_seed=0, benchmark_version="test")
