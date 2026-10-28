#!/usr/bin/env python3
"""Run QLoRA/LoRA SFT on prepared ToolForge data.

Examples:
    # Smoke test first (2 steps, 8 samples, CPU or GPU):
    python scripts/train_sft.py --config configs/sft.yaml --smoke

    # Full run (Colab T4):
    python scripts/train_sft.py --config configs/sft.yaml --use-4bit \
        --gradient-checkpointing --model-id Qwen/Qwen3-0.6B

    # Resume an interrupted run:
    python scripts/train_sft.py --config configs/sft.yaml --resume

Outputs under --output-dir (default outputs/checkpoints/sft):
    trainer/checkpoint-N/...  resumable Trainer checkpoints
    adapter-final/            LoRA adapter + tokenizer + sft_metadata.json
    run_metadata.json         settings, seeds, versions, data fingerprints, losses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.sft import SftSettings, load_sft_settings, run_sft  # noqa: E402

logger = get_logger("train_sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "sft.yaml"))
    p.add_argument("--smoke", action="store_true",
                   help="2 optimizer steps on <=8 samples; verifies the full stack cheaply")
    p.add_argument("--model-id", default=None)
    p.add_argument("--train-file", default=None)
    p.add_argument("--val-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--use-4bit", dest="use_4bit", action="store_true", default=None)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    overrides: dict = {}
    for flag, key in (("--model_id", "model_id"), ("--train_file", "train_file"),
                      ("--val_file", "val_file"), ("--output_dir", "output_dir"),
                      ("--epochs", "num_epochs"), ("--max_steps", "max_steps"),
                      ("--batch_size", "per_device_batch_size"),
                      ("--grad_accum", "grad_accum_steps"), ("--lr", "learning_rate"),
                      ("--max_seq_len", "max_seq_len"), ("--use_4bit", "use_4bit"),
                      ("--gradient_checkpointing", "gradient_checkpointing"),
                      ("--resume", "resume"), ("--seed", "seed")):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.smoke:
        overrides.update(max_steps=2, max_train_samples=8, max_val_samples=4,
                         logging_steps=1, eval_steps=2, save_steps=2, warmup_steps=0,
                         grad_accum_steps=1)
    settings: SftSettings = load_sft_settings(args.config, overrides)
    logger.info("SFT settings: %s", json.dumps({k: getattr(settings, k) for k in
                                                ("model_id", "max_seq_len", "use_4bit",
                                                 "per_device_batch_size", "grad_accum_steps",
                                                 "learning_rate", "max_steps", "seed")}))
    result = run_sft(settings)
    print("\n=== SFT COMPLETE ===")
    print(f"adapter:       {result.adapter_dir}")
    print(f"train loss:    {result.train_loss:.4f}")
    print(f"eval loss:     {result.eval_loss if result.eval_loss is not None else 'n/a'}")
    print(f"global step:   {result.global_step}")
    print(f"dataset:       {result.dataset_stats['train_samples']} train / "
          f"{result.dataset_stats['val_samples']} val samples "
          f"(dropped too long: {result.dataset_stats['train_dropped_too_long']}/"
          f"{result.dataset_stats['val_dropped_too_long']})")
    print(f"metadata:      {result.metadata_path}")
    print("\nNext: python scripts/evaluate.py --policy hf "
          f"--model-id {settings.model_id} --adapter-id {result.adapter_dir} --split val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
