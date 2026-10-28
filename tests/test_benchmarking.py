"""Tests for the inference benchmarking utilities (CPU, no model downloads)."""

from __future__ import annotations

import pytest

from toolforge.benchmarking import benchmark_run, format_benchmark_report, gpu_peak_memory_bytes
from toolforge.inference import HuggingFacePolicy, RandomPolicy
from toolforge.tasks import Task
from toolforge.tasks_builders import build_benchmark

from test_inference import FakeTokenizer  # reuse the injected-fake tokenizer


@pytest.fixture(scope="module")
def bench_tasks() -> list[Task]:
    tasks = build_benchmark(seed=20260301)
    by_id = {t.task_id: t for t in tasks}
    return [by_id[i] for i in ("tf-cal-ss-001", "tf-eml-ss-001", "tf-td-ss-001")]


class TestBenchmarkRun:
    def test_random_policy_measurements(self, bench_tasks, executor):
        bench = benchmark_run(bench_tasks, executor,
                              lambda ri, s: RandomPolicy(seed=3),
                              warmup_turns=1)
        assert bench["n_tasks"] == 3
        assert len(bench["episodes"]) == 3
        assert bench["model_load_seconds"] is None       # no model in RandomPolicy
        assert bench["warmup"]["turns"] == 1
        assert bench["turn_latency"]["n"] > 0
        assert bench["episode_latency"]["wall_seconds"]["median"] >= 0
        assert bench["gpu_peak_memory_bytes"] is None    # CPU-only sandbox
        assert bench["cost_estimate_per_1000_tasks"]["usd"] is None
        for ep in bench["episodes"]:
            assert ep["wall_seconds"] >= ep["policy_seconds"]
            assert ep["env_seconds"] >= 0.0

    def test_cost_estimate_math_is_explicit(self, bench_tasks, executor):
        rate = 0.42
        bench = benchmark_run(bench_tasks, executor, lambda ri, s: RandomPolicy(seed=4),
                              warmup_turns=0, gpu_cost_per_hour=rate,
                              hardware_label="fake rental GPU")
        cost = bench["cost_estimate_per_1000_tasks"]
        mean_wall = sum(e["wall_seconds"] for e in bench["episodes"]) / 3
        assert cost["usd"] == pytest.approx(mean_wall / 3600 * 1000 * rate, abs=1e-3)
        assert cost["assumed_hourly_rate_usd"] == rate
        assert cost["hardware_label"] == "fake rental GPU"
        assert "ESTIMATE" in cost["note"].upper() or "Estimate" in cost["note"]

    def test_hf_policy_with_fakes_measures_load(self, bench_tasks, executor):
        def fake_generate(prompt, kwargs):
            return '{"action": "final_answer", "content": "done"}', len(prompt.split()), 9

        def factory(ri, seed):
            return HuggingFacePolicy("fake/model", tokenizer=FakeTokenizer(),
                                     generate_fn=fake_generate, max_new_tokens=32)

        bench = benchmark_run(bench_tasks, executor, factory, warmup_turns=1,
                              measure_ttft=True)
        assert bench["model_load_seconds"] is not None
        # Non-streaming fallback path: TTFT is reported as unavailable, not faked.
        assert bench["ttft_seconds"] is None
        assert all(e["prompt_tokens"] and e["prompt_tokens"] > 0 for e in bench["episodes"])
        # Token totals are attributed per episode (warm-up and the TTFT probe are
        # deliberately excluded: they are not task work).
        assert bench["tokens"]["completion_total"] == 9 * sum(
            e["turns"] for e in bench["episodes"])

    def test_empty_tasks_rejected(self, executor):
        with pytest.raises(ValueError):
            benchmark_run([], executor, lambda ri, s: RandomPolicy(seed=1))

    def test_report_formatting(self, bench_tasks, executor):
        bench = benchmark_run(bench_tasks, executor, lambda ri, s: RandomPolicy(seed=5),
                              warmup_turns=1)
        text = format_benchmark_report(bench)
        assert "tasks benchmarked: 3" in text
        assert "cost per 1000 tasks: unavailable" in text
        assert "time-to-first-token: not measured" in text

    def test_gpu_memory_helper_cpu(self):
        assert gpu_peak_memory_bytes() is None
