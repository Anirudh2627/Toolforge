#!/usr/bin/env python3
"""Collect trajectories: reference solutions and/or policy rollouts (rejection sampled).

Examples:
    # Deterministic reference solutions for train+val (CPU, no model):
    python scripts/generate_trajectories.py --source reference

    # Self-generated rollouts of the base model on Colab (GPU):
    python scripts/generate_trajectories.py --source rollout --policy hf \
        --model-id Qwen/Qwen3-0.6B --samples-per-task 4 --do-sample

    # Optional teacher rollouts (requires the configured API key env var):
    python scripts/generate_trajectories.py --source rollout --policy api \
        --api-model gpt-4o-mini --source-label teacher --samples-per-task 2

Outputs: data/trajectories/<name>.jsonl + <name>_stats.json with rejection
rates and failure categories. Test-split tasks are refused by construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.config import load_config  # noqa: E402
from toolforge.data_generation import (  # noqa: E402
    collect_rollouts,
    generate_reference_trajectories,
    write_collection,
)
from toolforge.executor import Executor  # noqa: E402
from toolforge.inference import HuggingFacePolicy, OpenAICompatiblePolicy, RandomPolicy  # noqa: E402
from toolforge.registry import default_registry  # noqa: E402
from toolforge.tasks import load_tasks  # noqa: E402

logger = get_logger("generate_trajectories")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--tasks", default=None)
    p.add_argument("--source", choices=["reference", "rollout"], default="reference")
    p.add_argument("--splits", default="train,val", help="comma list; 'test' is always refused")
    p.add_argument("--policy", choices=["random", "hf", "api"], default="random",
                   help="rollout policy (--source rollout)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--adapter-id", default=None)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--api-model", default=None)
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--source-label", default=None, help="trajectory label: self|teacher|random")
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--max-kept-per-task", type=int, default=None)
    p.add_argument("--keep-rejections", action="store_true",
                   help="also persist rejected rollouts (verified=false) for analysis")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--out-dir", default="data/trajectories")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config if Path(args.config).exists() else None)
    tasks = load_tasks(args.tasks or config.paths.tasks_file)
    wanted = set(args.splits.split(","))
    if "test" in wanted:
        raise SystemExit("The test split is reserved for evaluation; refusing to collect on it.")
    pool = [t for t in tasks if t.split in wanted]
    logger.info("Collecting %s trajectories for %d tasks (splits=%s)",
                args.source, len(pool), sorted(wanted))

    executor = Executor(default_registry(), settings=config.env,
                        env_options={"sql": {"max_rows": config.env.sql_max_rows,
                                             "progress_limit": config.env.sql_progress_limit}})

    if args.source == "reference":
        trajs, stats = generate_reference_trajectories(
            pool, executor, tools_fingerprint="", run_tag="ref")
        name = args.name or "reference_trainval"
    else:
        label = args.source_label or {"hf": "self", "api": "teacher", "random": "random"}[args.policy]
        gen = dict(max_new_tokens=args.max_new_tokens)

        def factory(seed: int):
            if args.policy == "random":
                return RandomPolicy(seed=seed, max_tool_calls=4)
            if args.policy == "hf":
                return HuggingFacePolicy(
                    args.model_id or config.evaluation.model.model_id,
                    adapter_id=args.adapter_id,
                    device=config.evaluation.model.device, dtype=config.evaluation.model.dtype,
                    do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                    seed=seed, enable_thinking=config.evaluation.model.enable_thinking,
                    **gen)
            if not args.api_model:
                raise SystemExit("--api-model required for --policy api")
            return OpenAICompatiblePolicy(
                args.api_model, base_url=args.api_base_url or "https://api.openai.com/v1",
                api_key_env=args.api_key_env, temperature=args.temperature, **gen)

        trajs, stats = collect_rollouts(
            pool, executor, factory, samples_per_task=args.samples_per_task,
            source=label, base_seed=args.base_seed, max_kept_per_task=args.max_kept_per_task,
            keep_rejections=args.keep_rejections)
        name = args.name or f"rollouts_{label}_{args.policy}"

    path = write_collection(args.out_dir, name, trajs, stats)
    print(f"\nwrote {len(trajs)} trajectories -> {path}")
    print(f"attempted={stats.attempted} kept={stats.kept} rejected={stats.rejected} "
          f"uncovered={len(stats.uncovered)}")
    if stats.failure_categories:
        print("top failure categories:", stats.failure_categories.most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
