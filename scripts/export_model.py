#!/usr/bin/env python3
"""Export a standalone model: merge a LoRA adapter into its base weights.

Example:
    python scripts/export_model.py \
        --adapter outputs/checkpoints/sft/adapter-final \
        --base-model Qwen/Qwen3-0.6B \
        --out outputs/merged/sft-0.6b

Licensing: the merged directory contains base-model weights and is subject to
the BASE MODEL's license (not this repo's MIT license). Check the model card
before redistributing or publishing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge.checkpointing import export_merged_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = p.parse_args()
    out = export_merged_model(args.adapter, args.base_model, args.out, dtype=args.dtype)
    print(f"Merged model exported to {out}")
    print("Reminder: redistribution of the merged weights follows the BASE MODEL's license.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
