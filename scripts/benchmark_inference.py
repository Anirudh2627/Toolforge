#!/usr/bin/env python3
"""Benchmark inference latency, throughput, tokens, and (optionally) cost.

Examples:
    # CPU smoke benchmark with the random baseline:
    python scripts/benchmark_inference.py --policy random --n-tasks 6

    # Real model benchmark on GPU, with TTFT and a cost estimate based on the
    # hourly rate YOU supply (the script never invents prices):
    python scripts/benchmark_inference.py --policy hf --model-id Qwen/Qwen3-0.6B \
        --n-tasks 12 --ttft --gpu-cost-per-hour 0.35 --hardware-label "rented T4 (vast.ai)"

Writes a JSON measurement file under outputs/benchmark/ and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.benchmarking import benchmark_run, format_benchmark_report  # noqa: E402
from toolforge.config import load_config  # noqa: E402
from toolforge.evaluation import filter_tasks  # noqa: E402
from toolforge.executor import Executor  # noqa: E402
from toolforge.inference import HuggingFacePolicy, RandomPolicy  # noqa: E402
from toolforge.registry import default_registry  # noqa: E402
from toolforge.tasks import load_tasks  # noqa: E402

logger = get_logger("benchmark_inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "evaluation.yaml"))
    p.add_argument("--tasks", default=None)
    p.add_argument("--policy", choices=["random", "hf"], default="random")
    p.add_argument("--model-id", default=None)
    p.add_argument("--adapter-id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--do-sample", action="store_true", default=None)
    p.add_argument("--n-tasks", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=1234,
                   help="Seed for the deterministic task subsample")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="Policy/sampling seed")
    p.add_argument("--ttft", action="store_true", help="Measure time-to-first-token (hf only)")
    p.add_argument("--gpu-cost-per-hour", type=float, default=None,
                   help="Hourly rate you pay/assume; required for any cost number")
    p.add_argument("--hardware-label", default=None,
                   help="e.g. 'Colab free T4' — recorded with the cost estimate")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config if Path(args.config).exists() else None)
    tasks_path = Path(args.tasks or config.paths.tasks_file)
    all_tasks = load_tasks(tasks_path)

    # Deterministic subsample (benchmarking, not accuracy: any split is fine,
    # but the selection must be reproducible).
    pool = filter_tasks(all_tasks, split="all")
    rng = random.Random(args.sample_seed)
    tasks = sorted(rng.sample(pool, min(args.n_tasks, len(pool))), key=lambda t: t.task_id)

    gen = config.evaluation.generation
    model_cfg = config.evaluation.model
    if args.policy == "hf":
        def factory(run_index: int, sampling_seed: int):
            return HuggingFacePolicy(
                args.model_id or model_cfg.model_id,
                adapter_id=args.adapter_id or model_cfg.adapter_id,
                device=args.device or model_cfg.device,
                dtype=args.dtype or model_cfg.dtype,
                max_new_tokens=args.max_new_tokens or gen.max_new_tokens,
                do_sample=gen.do_sample if args.do_sample is None else args.do_sample,
                seed=sampling_seed, enable_thinking=model_cfg.enable_thinking)
    else:
        def factory(run_index: int, sampling_seed: int):
            return RandomPolicy(seed=args.seed + run_index)

    executor = Executor(default_registry(), settings=config.env,
                        env_options={"sql": {"max_rows": config.env.sql_max_rows,
                                             "progress_limit": config.env.sql_progress_limit}})

    bench = benchmark_run(
        tasks, executor, factory,
        warmup_turns=args.warmup, sampling_seed=args.seed, measure_ttft=args.ttft,
        gpu_cost_per_hour=args.gpu_cost_per_hour, hardware_label=args.hardware_label)
    bench["selection"] = {"sample_seed": args.sample_seed,
                          "task_ids": [t.task_id for t in tasks],
                          "tasks_file": str(tasks_path)}
    bench["config"] = {"max_new_tokens": args.max_new_tokens or gen.max_new_tokens,
                       "do_sample": bench_policy_do_sample(args, gen)}

    out_path = Path(args.out) if args.out else \
        Path(config.paths.output_dir) / "benchmark" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(format_benchmark_report(bench))
    print(f"\nmeasurements written to {out_path}")
    return 0


def bench_policy_do_sample(args: argparse.Namespace, gen) -> bool:
    return gen.do_sample if args.do_sample is None else args.do_sample


if __name__ == "__main__":
    raise SystemExit(main())
