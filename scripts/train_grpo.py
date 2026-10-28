#!/usr/bin/env python3
"""Run GRPO (verifier-backed rewards) on top of an SFT checkpoint.

Examples:
    # Compatibility probe first (once per machine):
    python scripts/check_train_env.py

    # Smoke test (tiny budget, verifies the full loop cheaply):
    python scripts/train_grpo.py --config configs/grpo.yaml --smoke \
        --adapter-id outputs/checkpoints/sft/adapter-final

    # Real run on Colab (resumable):
    python scripts/train_grpo.py --config configs/grpo.yaml \
        --adapter-id outputs/checkpoints/sft/adapter-final --compact-catalog

    # Resume after a session drop:
    python scripts/train_grpo.py --config configs/grpo.yaml \
        --adapter-id outputs/checkpoints/sft/adapter-final --resume

Outputs under --output-dir: checkpoint-iterNNNNN/ (adapter + optimizer + RNG +
metadata), history.jsonl (per-iteration rewards/success/eval), run_metadata.json.
Evaluate the result with:
    python scripts/evaluate.py --policy hf --model-id <base> \
        --adapter-id <output-dir>/checkpoint-iterNNNNN --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.grpo import GrpoTrainer, load_grpo_settings  # noqa: E402

logger = get_logger("train_grpo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "grpo.yaml"))
    p.add_argument("--smoke", action="store_true",
                   help="2 iterations, G=2, 2 tasks, eval on 2 tasks — stack check only")
    p.add_argument("--model-id", default=None)
    p.add_argument("--adapter-id", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tasks-file", default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--rollout-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--beta-kl", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--compact-catalog", action="store_true", default=None)
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument("--use-4bit", action="store_true", default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {
        "model_id": args.model_id, "adapter_id": args.adapter_id,
        "output_dir": args.output_dir, "tasks_file": args.tasks_file,
        "total_iterations": args.iterations, "num_generations": args.num_generations,
        "rollout_batch_size": args.rollout_batch_size, "learning_rate": args.lr,
        "beta_kl": args.beta_kl, "max_seq_len": args.max_seq_len,
        "compact_catalog": args.compact_catalog,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_4bit": args.use_4bit, "resume": args.resume, "seed": args.seed,
        "eval_every": args.eval_every, "save_every": args.save_every,
    }
    if args.smoke:
        overrides.update({k: v for k, v in {
            "total_iterations": 2, "num_generations": 2, "rollout_batch_size": 2,
            "task_limit": 2, "eval_every": 2, "eval_task_limit": 2, "save_every": 2,
            "warmup_steps": 0, "max_new_tokens": 128,
        }.items() if overrides.get(k) is None})
    settings = load_grpo_settings(args.config, overrides)
    logger.info("GRPO settings: %s", json.dumps(
        {k: getattr(settings, k) for k in
         ("model_id", "adapter_id", "num_generations", "rollout_batch_size",
          "total_iterations", "learning_rate", "beta_kl", "compact_catalog", "seed")}))
    trainer = GrpoTrainer(settings)
    summary = trainer.train()
    print("\n=== GRPO COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print("\nNext: independent evaluation of the final checkpoint:")
    print(f"  python scripts/evaluate.py --policy hf --model-id {settings.model_id} "
          f"--adapter-id {summary['final_checkpoint']} --split test"
          + (" --compact-catalog" if settings.compact_catalog else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
