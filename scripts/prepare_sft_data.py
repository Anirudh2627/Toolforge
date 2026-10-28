#!/usr/bin/env python3
"""Prepare SFT data: mix trajectory sources -> messages-level JSONL (tokenizer-free).

Examples:
    python scripts/prepare_sft_data.py \
        --trajectories data/trajectories/reference_trainval.jsonl \
        --out-dir data/sft --max-per-task 3

    # Mix in self-generated rollouts (reference data is kept first):
    python scripts/prepare_sft_data.py \
        --trajectories data/trajectories/reference_trainval.jsonl \
                   data/trajectories/rollouts_self_hf.jsonl \
        --out-dir data/sft --max-per-task 3

Writes sft_train.jsonl / sft_val.jsonl (records = {"messages": [...]} + provenance),
plus sft_stats.json. Test-split and held-out-family trajectories are refused.
Tokenization happens in scripts/train_sft.py (needs the real tokenizer).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.config import load_config  # noqa: E402
from toolforge.data_generation import build_sft_mix  # noqa: E402
from toolforge.sft_data import build_sft_records, save_sft_records  # noqa: E402
from toolforge.trajectories import load_trajectories  # noqa: E402

logger = get_logger("prepare_sft_data")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--trajectories", nargs="+", required=True,
                   help="trajectory JSONL files, in priority order (earlier = kept first)")
    p.add_argument("--out-dir", default="data/sft")
    p.add_argument("--max-per-task", type=int, default=None)
    p.add_argument("--allow-dirty", action="store_true",
                   help="also keep trajectories containing invalid calls (default: clean only)")
    args = p.parse_args()

    config = load_config(args.config if Path(args.config).exists() else None)
    sources = [load_trajectories(path) for path in args.trajectories]
    reference, *others = sources if sources else ([], )
    pool, mix_stats = build_sft_mix(
        reference, *others,
        held_out_families=config.benchmark.held_out_families,
        max_per_task=args.max_per_task)

    records, filter_stats = build_sft_records(
        pool, require_verified=True, require_clean=not args.allow_dirty)

    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    if not train_records:
        raise SystemExit("No train-split SFT records; run scripts/generate_trajectories.py first")

    out = Path(args.out_dir)
    save_sft_records(out / "sft_train.jsonl", train_records)
    if val_records:
        save_sft_records(out / "sft_val.jsonl", val_records)

    stats = {
        "mix": mix_stats,
        "record_filter": filter_stats,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "by_category": dict(Counter(r["category"] for r in train_records)),
        "by_family": dict(Counter(r["family"] for r in train_records)),
        "by_source": dict(Counter(r["source"] for r in train_records)),
        "inputs": [str(p) for p in args.trajectories],
    }
    (out / "sft_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    logger.info("Wrote %d train / %d val records -> %s", len(train_records), len(val_records), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
