"""Inference benchmarking: latency, throughput, token, and cost measurement.

Measurement rules (research integrity):

- Warm-up generations run before any timing is recorded.
- Model inference time (``policy_seconds``) and CPU environment time
  (``env_seconds``) are reported separately; wall time is their sum.
- Median and p95 are reported whenever the sample size supports them (n is
  always included so readers can judge).
- Peak GPU memory is reported only when torch+CUDA are actually in use.
- Cost per 1,000 tasks is computed ONLY from measured wall time and an
  explicitly supplied hourly rate, and is labeled as an estimate with its
  basis and hardware string. No default rates, no invented numbers.
"""

from __future__ import annotations

import time
from typing import Any

from toolforge import get_logger
from toolforge.evaluation import PolicyFactory, make_context_stub
from toolforge.executor import Executor
from toolforge.metrics import latency_summary, percentile
from toolforge.tasks import Task
from toolforge.verifier import verify_episode

logger = get_logger("benchmarking")


def gpu_peak_memory_bytes() -> int | None:
    """Peak allocated CUDA bytes for this process, or None when unavailable."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:
        pass
    return None


def warmup(policy: Any, task: Task, executor: Executor, turns: int = 3) -> float:
    """Run ``turns`` untimed-context generations; returns total warm-up seconds."""
    context = make_context_stub(task, executor)
    t0 = time.perf_counter()
    for _ in range(turns):
        policy(context)
    elapsed = time.perf_counter() - t0
    logger.info("Warm-up: %d generations in %.2fs", turns, elapsed)
    return elapsed


def benchmark_run(tasks: list[Task], executor: Executor, policy_factory: PolicyFactory, *,
                  warmup_turns: int = 3, sampling_seed: int = 0, measure_ttft: bool = False,
                  gpu_cost_per_hour: float | None = None,
                  hardware_label: str | None = None) -> dict[str, Any]:
    """Benchmark one policy over ``tasks`` (full episodes, verified).

    Returns a measurement dict; every timing is measured in this process.
    """
    if not tasks:
        raise ValueError("benchmark_run requires at least one task")
    policy = policy_factory(0, sampling_seed)

    result: dict[str, Any] = {
        "policy_id": getattr(policy, "policy_id", type(policy).__name__),
        "n_tasks": len(tasks),
        "hardware_label": hardware_label,
        "model_load_seconds": None,
        "warmup": None,
        "ttft_seconds": None,
        "episodes": [],
    }

    # Load + warm-up (untimed for the measurement itself).
    loader = getattr(policy, "_load", None)
    if callable(loader):
        t0 = time.perf_counter()
        loader()
        result["model_load_seconds"] = round(
            getattr(policy, "load_seconds", None) or (time.perf_counter() - t0), 4)
    result["warmup"] = {"turns": warmup_turns,
                        "seconds": round(warmup(policy, tasks[0], executor, warmup_turns), 4)}

    if measure_ttft:
        ttft_fn = getattr(policy, "generate_with_ttft", None)
        if callable(ttft_fn):
            context = make_context_stub(tasks[0], executor)
            _, stats = ttft_fn(context)
            result["ttft_seconds"] = stats.ttft_seconds
        else:
            logger.warning("measure_ttft requested but policy %s does not support streaming; "
                           "reporting null rather than a substitute number",
                           result["policy_id"])

    turn_latencies: list[float] = []
    for task in tasks:
        reset = getattr(policy, "reset_stats", None)
        if callable(reset):
            reset()
        t_wall0 = time.perf_counter()
        episode = executor.run(task, policy)
        wall = time.perf_counter() - t_wall0
        verdict = verify_episode(task, episode)
        turn_latencies.extend(s.policy_seconds for s in episode.trace)
        stats = getattr(policy, "stats", None)
        result["episodes"].append({
            "task_id": task.task_id,
            "success": verdict.success,
            "termination": episode.termination,
            "tool_call_count": episode.tool_call_count,
            "invalid_call_count": episode.invalid_call_count,
            "turns": len(episode.trace),
            "final_output_chars": len(episode.final_output or ""),
            "policy_seconds": round(episode.policy_seconds, 6),
            "env_seconds": round(episode.env_seconds, 6),
            "wall_seconds": round(wall, 6),
            "prompt_tokens": getattr(stats, "prompt_tokens", None),
            "completion_tokens": getattr(stats, "completion_tokens", None),
        })

    eps = result["episodes"]
    successes = sum(1 for e in eps if e["success"])
    total_wall = sum(e["wall_seconds"] for e in eps)
    result["turn_latency"] = latency_summary(turn_latencies)
    result["episode_latency"] = {
        "policy_seconds": latency_summary([e["policy_seconds"] for e in eps]),
        "env_seconds": latency_summary([e["env_seconds"] for e in eps]),
        "wall_seconds": latency_summary([e["wall_seconds"] for e in eps]),
    }
    result["throughput"] = {
        "tasks_per_second": len(eps) / total_wall if total_wall > 0 else None,
        "successful_tasks_per_second": successes / total_wall if total_wall > 0 else None,
        "success_rate_on_benchmark_subset": successes / len(eps),
        "total_wall_seconds": round(total_wall, 4),
    }
    prompt_tokens = [e["prompt_tokens"] for e in eps if e["prompt_tokens"] is not None]
    completion_tokens = [e["completion_tokens"] for e in eps if e["completion_tokens"] is not None]
    result["tokens"] = {
        "prompt_total": sum(prompt_tokens) if prompt_tokens else None,
        "completion_total": sum(completion_tokens) if completion_tokens else None,
        "prompt_mean_per_task": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
        "completion_mean_per_task": (sum(completion_tokens) / len(completion_tokens))
        if completion_tokens else None,
        "tool_calls_per_task_mean": sum(e["tool_call_count"] for e in eps) / len(eps),
        "final_output_chars_p95": percentile([e["final_output_chars"] for e in eps], 0.95),
    }
    result["gpu_peak_memory_bytes"] = gpu_peak_memory_bytes()

    # Cost: only from measured wall time + an explicitly supplied hourly rate.
    if gpu_cost_per_hour is not None:
        mean_wall = total_wall / len(eps)
        cost_1000 = mean_wall / 3600.0 * 1000 * gpu_cost_per_hour
        result["cost_estimate_per_1000_tasks"] = {
            "usd": round(cost_1000, 4),
            "basis": "measured_mean_wall_time_x_supplied_hourly_rate",
            "assumed_hourly_rate_usd": gpu_cost_per_hour,
            "hardware_label": hardware_label or "unspecified (state it before quoting this number)",
            "mean_wall_seconds_per_task": round(mean_wall, 4),
            "note": "Estimate, not a measurement of price. Rate was supplied by the user.",
        }
    else:
        result["cost_estimate_per_1000_tasks"] = {
            "usd": None,
            "basis": "unavailable",
            "note": "Pass --gpu-cost-per-hour with the rate you actually pay/assume to get "
                    "an explicitly labeled estimate.",
        }
    return result


def format_benchmark_report(bench: dict[str, Any]) -> str:
    """Human-readable summary of benchmark_run output (for CLI/logs)."""
    lines = [
        f"policy: {bench['policy_id']}",
        f"tasks benchmarked: {bench['n_tasks']}",
        f"model load: {bench['model_load_seconds']} s",
        f"warm-up: {bench['warmup']['turns']} generations in {bench['warmup']['seconds']} s",
        f"time-to-first-token: {bench['ttft_seconds']} s" if bench["ttft_seconds"] is not None
        else "time-to-first-token: not measured (policy lacks streaming support)",
        f"turn latency (policy): median {bench['turn_latency']['median']:.3f}s "
        f"p95 {bench['turn_latency']['p95']:.3f}s (n={bench['turn_latency']['n']})"
        if bench["turn_latency"]["n"] else "turn latency: no turns recorded",
        f"episode wall: median {bench['episode_latency']['wall_seconds']['median']:.3f}s "
        f"p95 {bench['episode_latency']['wall_seconds']['p95']:.3f}s",
        f"env-only time per episode (median): "
        f"{bench['episode_latency']['env_seconds']['median']:.4f}s",
        f"throughput: {bench['throughput']['tasks_per_second']:.3f} tasks/s "
        f"({bench['throughput']['successful_tasks_per_second']:.3f} successful/s)",
        f"peak GPU memory: {bench['gpu_peak_memory_bytes']} bytes"
        if bench["gpu_peak_memory_bytes"] is not None else "peak GPU memory: unavailable (no CUDA)",
    ]
    cost = bench["cost_estimate_per_1000_tasks"]
    if cost["usd"] is not None:
        lines.append(f"cost per 1000 tasks (ESTIMATE, rate=${cost['assumed_hourly_rate_usd']}/h "
                     f"on '{cost['hardware_label']}'): ${cost['usd']}")
    else:
        lines.append("cost per 1000 tasks: unavailable (no hourly rate supplied)")
    return "\n".join(lines)
