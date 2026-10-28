#!/usr/bin/env python3
"""Generate the ToolForge task benchmark (deterministic, CPU-only).

Usage:
    python scripts/generate_tasks.py --out data/tasks/benchmark_v1.jsonl
    python scripts/generate_tasks.py --config configs/base.yaml --summary-only

Outputs:
    <out>                 one JSON task per line (JSONL)
    <out without .jsonl>_summary.json   category/family/split counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.config import load_config  # noqa: E402
from toolforge.tasks import benchmark_summary, check_no_leakage, save_tasks  # noqa: E402
from toolforge.tasks_builders import build_benchmark  # noqa: E402

logger = get_logger("generate_tasks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    parser.add_argument("--seed", type=int, default=None, help="Override benchmark seed")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: from config)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary to stdout without writing files")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.benchmark.seed
    out_path = Path(args.out) if args.out else Path(config.paths.tasks_file)
    if args.summary_only:
        out_path = None

    tasks = build_benchmark(
        seed=seed,
        held_out_families=config.benchmark.held_out_families,
        train_fraction=config.benchmark.train_fraction,
        val_fraction=config.benchmark.val_fraction,
        version=config.benchmark.version,
    )
    # build_benchmark already checks leakage; re-check to keep this script honest
    # when invoked with overridden seeds/fractions.
    check_no_leakage(tasks, config.benchmark.held_out_families)

    summary = benchmark_summary(tasks)
    summary["benchmark_seed"] = seed
    summary["benchmark_version"] = config.benchmark.version
    summary["held_out_families"] = list(config.benchmark.held_out_families)

    print(json.dumps(summary, indent=2))

    if out_path is not None:
        n = save_tasks(out_path, tasks)
        summary_path = out_path.with_suffix("").parent / (out_path.stem + "_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %d tasks to %s (summary: %s)", n, out_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
