#!/usr/bin/env python3
"""Evaluate a policy on the ToolForge benchmark (CPU or GPU, any policy).

Examples:
    # CPU smoke test with the random baseline (no model download):
    python scripts/evaluate.py --policy random --split test --limit 8 \
        --run-name random-smoke

    # Full base-model evaluation (needs torch+transformers, GPU recommended):
    python scripts/evaluate.py --policy hf --model-id Qwen/Qwen3-0.6B \
        --split test --run-name base-0.6b

    # SFT/GRPO adapter evaluation:
    python scripts/evaluate.py --policy hf --model-id Qwen/Qwen3-1.7B \
        --adapter-id outputs/checkpoints/sft-final --split test

    # Optional external teacher (requires the API key env var you configure):
    python scripts/evaluate.py --policy api --api-model gpt-4o-mini --split test

Outputs (under --out/<run-name>/): predictions.jsonl (per-task, resumable),
metrics.json, metrics.csv, run_config.json. Metrics are recomputable from
predictions alone: see toolforge.evaluation.reaggregate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.config import config_to_dict, load_config  # noqa: E402
from toolforge.evaluation import RunMetadata, filter_tasks, run_evaluation  # noqa: E402
from toolforge.executor import Executor  # noqa: E402
from toolforge.inference import (  # noqa: E402
    HuggingFacePolicy,
    OpenAICompatiblePolicy,
    RandomPolicy,
)
from toolforge.registry import default_registry  # noqa: E402
from toolforge.tasks import load_tasks  # noqa: E402

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "evaluation.yaml"))
    p.add_argument("--tasks", default=None, help="Task JSONL (default: config paths.tasks_file)")
    p.add_argument("--policy", choices=["random", "hf", "api"], default="random")
    p.add_argument("--policy-seed", type=int, default=0)
    # hf policy
    p.add_argument("--model-id", default=None)
    p.add_argument("--adapter-id", default=None)
    p.add_argument("--device", default=None, help="auto|cpu|cuda (default: config)")
    p.add_argument("--dtype", default=None, help="auto|float16|bfloat16|float32")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--do-sample", action="store_true", default=None)
    p.add_argument("--greedy", dest="do_sample", action="store_false")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--compact-catalog", action="store_true", default=None,
                   help="Render the system-prompt tool catalog compactly. MUST match how "
                        "the evaluated checkpoint was trained (prepare_sft_data --catalog).")
    # api policy
    p.add_argument("--api-model", default=None)
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    # selection / run control
    p.add_argument("--split", default=None, help="train|val|test|all (default: config)")
    p.add_argument("--family", action="append", default=None)
    p.add_argument("--category", action="append", default=None)
    p.add_argument("--task-id", action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Base sampling seed")
    p.add_argument("--out", default=None, help="Output root dir (default: config evaluation.output_dir)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-states", action="store_true",
                   help="Do not persist initial/final env states (smaller files, less reproducible)")
    return p.parse_args()


def build_policy_factory(args: argparse.Namespace, config) -> tuple[object, dict, dict]:
    """Return (policy_factory, generation_meta, model_meta)."""
    gen_cfg = config.evaluation.generation
    model_cfg = config.evaluation.model
    generation = {
        "max_new_tokens": args.max_new_tokens or gen_cfg.max_new_tokens,
        "do_sample": gen_cfg.do_sample if args.do_sample is None else args.do_sample,
        "temperature": args.temperature if args.temperature is not None else gen_cfg.temperature,
        "top_p": args.top_p if args.top_p is not None else gen_cfg.top_p,
    }

    if args.policy == "random":
        def factory(run_index: int, sampling_seed: int) -> RandomPolicy:
            return RandomPolicy(seed=args.policy_seed + run_index)
        return factory, generation, {"policy": "random", "policy_seed": args.policy_seed}

    if args.policy == "hf":
        model_meta = {
            "policy": "hf",
            "model_id": args.model_id or model_cfg.model_id,
            "adapter_id": args.adapter_id or model_cfg.adapter_id,
            "device": args.device or model_cfg.device,
            "dtype": args.dtype or model_cfg.dtype,
            "enable_thinking": model_cfg.enable_thinking,
            "compact_catalog": (model_cfg.compact_catalog
                                if args.compact_catalog is None else args.compact_catalog),
        }
        state: dict = {"policy": None}

        def factory(run_index: int, sampling_seed: int) -> HuggingFacePolicy:
            if state["policy"] is None:  # load the model once, reuse across runs
                state["policy"] = HuggingFacePolicy(
                    model_meta["model_id"], adapter_id=model_meta["adapter_id"],
                    device=model_meta["device"], dtype=model_meta["dtype"],
                    max_new_tokens=generation["max_new_tokens"],
                    do_sample=generation["do_sample"],
                    temperature=generation["temperature"], top_p=generation["top_p"],
                    seed=sampling_seed, enable_thinking=model_meta["enable_thinking"],
                    compact_catalog=model_meta["compact_catalog"])
            else:
                state["policy"].set_sampling(seed=sampling_seed)
            return state["policy"]
        return factory, generation, model_meta

    if args.policy == "api":
        if not args.api_model:
            raise SystemExit("--api-model is required for --policy api")
        meta = {"policy": "api", "model": args.api_model,
                "base_url": args.api_base_url or "https://api.openai.com/v1",
                "api_key_env": args.api_key_env}

        def factory(run_index: int, sampling_seed: int) -> OpenAICompatiblePolicy:
            return OpenAICompatiblePolicy(
                args.api_model,
                base_url=meta["base_url"], api_key_env=args.api_key_env,
                max_new_tokens=generation["max_new_tokens"],
                temperature=generation["temperature"])
        return factory, generation, meta

    raise SystemExit(f"Unknown policy {args.policy!r}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config if Path(args.config).exists() else None)
    ev = config.evaluation

    tasks_path = Path(args.tasks or config.paths.tasks_file)
    all_tasks = load_tasks(tasks_path)
    tasks = filter_tasks(
        all_tasks,
        split=args.split or ev.split,
        families=args.family, categories=args.category, task_ids=args.task_id,
        limit=args.limit if args.limit is not None else ev.task_limit)

    policy_factory, generation, model_meta = build_policy_factory(args, config)
    registry = default_registry()
    executor = Executor(registry, settings=config.env,
                        env_options={"sql": {"max_rows": config.env.sql_max_rows,
                                             "progress_limit": config.env.sql_progress_limit}})

    run_name = args.run_name or f"{model_meta.get('policy', 'policy')}-{args.split or ev.split}-" \
                                f"{time.strftime('%Y%m%d-%H%M%S')}"
    run_name = run_name.replace("/", "_")
    out_root = Path(args.out or ev.output_dir)
    out_dir = out_root / run_name

    metadata = RunMetadata(
        run_name=run_name,
        policy_description=f"{model_meta.get('policy')}"
                           + (f":{model_meta.get('model_id')}" if model_meta.get("model_id") else "")
                           + (f"+adapter:{model_meta.get('adapter_id')}"
                              if model_meta.get("adapter_id") else ""),
        task_file=str(tasks_path),
        split=args.split or ev.split,
        n_tasks=len(tasks),
        repeat=args.repeat or ev.repeat,
        base_seed=args.seed if args.seed is not None else ev.seed,
        generation=generation,
        model=model_meta,
        config_snapshot=config_to_dict(config),
        extra={"argv": sys.argv[1:]},
    )

    logger.info("Evaluating %d tasks (split=%s, repeat=%d) with policy %s -> %s",
                len(tasks), metadata.split, metadata.repeat, metadata.policy_description, out_dir)
    summary = run_evaluation(
        tasks, executor, policy_factory,
        out_dir=out_dir, run_name=run_name, run_metadata=metadata,
        repeat=metadata.repeat, base_seed=metadata.base_seed,
        include_states=not args.no_states and ev.include_states,
        resume=not args.no_resume, progress_every=ev.progress_every,
        bootstrap_iterations=ev.bootstrap_iterations)

    ci = summary.get("wilson_ci") or {}
    boot = summary.get("bootstrap_ci") or {}
    wilson_str = f"wilson95 [{ci['low']:.3f}, {ci['high']:.3f}]" if ci else "no CI"
    boot_str = (f"bootstrap95 [{boot['low']:.3f}, {boot['high']:.3f}]" if boot else "no bootstrap")
    print("\n=== RESULTS ===")
    print(f"tasks: {summary['n_records']} records / {summary['n_tasks']} tasks")
    print(f"success rate: {summary['success_rate']:.3f} ({wilson_str}; {boot_str})")
    print(f"held-out family success: {summary['held_out_family_success']}")
    print(f"invalid_json_rate: {summary['invalid_json_rate']:.3f} | "
          f"invalid_tool_call_rate: {summary['invalid_tool_call_rate']:.3f} | "
          f"arg_validation_failure_rate: {summary['argument_validation_failure_rate']:.3f}")
    print(f"step_limit_violation_rate: {summary['step_limit_violation_rate']:.3f} | "
          f"loop_guard_rate: {summary['loop_guard_rate']:.3f} | "
          f"unwanted_side_effect_rate: {summary['unwanted_side_effect_rate']:.3f}")
    print(f"clarification: P={summary['clarification']['precision']} "
          f"R={summary['clarification']['recall']} F1={summary['clarification']['f1']}")
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
