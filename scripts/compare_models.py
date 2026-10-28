#!/usr/bin/env python3
"""Compare evaluation runs: tables, paired analysis, plots, REPORT.md.

Examples:
    # The standard three-way ablation (runs from scripts/evaluate.py):
    python scripts/compare_models.py \
        --run base=outputs/eval/base-0.6b \
        --run sft=outputs/eval/sft-0.6b \
        --run sft-grpo=outputs/eval/grpo-0.6b \
        --history sft-grpo=outputs/checkpoints/grpo/history.jsonl \
        --out reports/ablations

    # Data-amount ablation (SFT trained on different trajectory subsets):
    python scripts/compare_models.py --run sft-25=eval/run-a --run sft-50=eval/run-b \
        --run sft-100=eval/run-c --out reports/ablations-data

    # Repeated-training variability (independently trained runs, same config):
    python scripts/compare_models.py ... --group sft=eval/sft-seed1,eval/sft-seed2

    # Documentation preview ONLY (clearly watermarked fake data):
    python scripts/compare_models.py --illustrative --out /tmp/preview

Outputs in --out: comparison.json, comparison.csv, paired.json, REPORT.md,
plots/*.png, provenance.json (input paths + fingerprints). Exits non-zero when
runs are not comparable (grammar/fingerprint/split mismatch or metrics that do
not match their predictions) unless --allow-incompatible is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge import get_logger  # noqa: E402
from toolforge.comparison import (  # noqa: E402
    check_compatibility,
    comparison_table,
    comparison_to_csv,
    find_regressions,
    load_history,
    load_run,
    paired_analysis,
    repeat_group_stats,
    report_markdown,
    rl_curve_data,
)

logger = get_logger("compare_models")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                   help="evaluation run directory (repeatable; order = pipeline order)")
    p.add_argument("--history", action="append", default=[], metavar="LABEL=PATH",
                   help="GRPO history.jsonl for RL-dynamics figures (repeatable)")
    p.add_argument("--group", action="append", default=[], metavar="LABEL=P1,P2",
                   help="repeat-group: independently trained runs of one configuration")
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "ablations"))
    p.add_argument("--regression-threshold", type=float, default=0.05)
    p.add_argument("--allow-incompatible", action="store_true",
                   help="compare anyway; every violation is recorded in REPORT.md")
    p.add_argument("--illustrative", action="store_true",
                   help="render watermarked fake-data previews (documentation only)")
    return p.parse_args()


def _split_kv(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--{what} expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        out[label.strip()] = path.strip()
    return out


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from toolforge.plots import illustrative_table, plot_all

    if args.illustrative:
        table = illustrative_table()
        written = plot_all(table, out / "plots", illustrative=True)
        (out / "comparison.json").write_text(json.dumps(
            {"ILLUSTRATIVE": True,
             "note": "Fake data for layout preview only. Not experimental results.",
             "table": table}, indent=2) + "\n", encoding="utf-8")
        print(f"Illustrative previews written to {out}/plots ({len(written)} files).")
        print("These figures contain FAKE, watermarked data — never report them as results.")
        return 0

    runs = _split_kv(args.run, "run")
    if len(runs) < 2:
        raise SystemExit("Provide at least two --run LABEL=PATH entries "
                         "(or use --illustrative for a layout preview)")

    bundles = [load_run(label, path) for label, path in runs.items()]
    warnings = check_compatibility(bundles, allow_incompatible=args.allow_incompatible)
    table = comparison_table(bundles)

    paired = [paired_analysis(bundles[i], bundles[i + 1])
              for i in range(len(bundles) - 1)]

    groups = []
    for spec in args.group:
        if "=" not in spec:
            raise SystemExit(f"--group expects LABEL=P1,P2, got {spec!r}")
        glabel, paths = spec.split("=", 1)
        gbundles = [load_run(f"{glabel}-r{i + 1}", p) for i, p in enumerate(paths.split(","))]
        groups.append(repeat_group_stats(glabel, gbundles))

    histories = {}
    for label, path in _split_kv(args.history, "history").items():
        histories[label] = rl_curve_data(load_history(path))

    regressions = find_regressions(bundles, table, threshold=args.regression_threshold)
    report = report_markdown(bundles, table, paired, groups, histories, warnings, regressions)

    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "warnings": warnings, "table": table, "paired": paired,
         "repeat_groups": groups, "regressions": regressions,
         "rl_histories": {k: {kk: vv for kk, vv in v.items() if kk != "iterations"}
                          for k, v in histories.items()}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "comparison.csv").write_text(comparison_to_csv(table), encoding="utf-8")
    (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "runs": {b.label: {"path": str(b.path),
                           "metrics_md5": hashlib.sha256(
                               (b.path / "metrics.json").read_bytes()).hexdigest()[:16],
                           "predictions_md5": hashlib.sha256(
                               (b.path / "predictions.jsonl").read_bytes()).hexdigest()[:16]}
                 for b in bundles},
        "histories": dict(_split_kv(args.history, "history")),
        "groups": args.group,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                         encoding="utf-8")

    written = plot_all(table, out / "plots", histories=histories or None)

    print(f"Compared {len(bundles)} runs -> {out}")
    for b in bundles:
        row = table[b.label]
        held = (row.get("held_out") or {}).get("success_rate")
        held_str = f"{held:.3f}" if held is not None else "n/a"
        print(f"  {b.label:>16s}: success {row['success_rate']:.3f} "
              f"(n={row['n_tasks']}, held-out {held_str})")
    for p in paired:
        print(f"  paired {p['a']} vs {p['b']}: only-{p['a']} {p['only_a_solved']}, "
              f"only-{p['b']} {p['only_b_solved']}, McNemar p={p['mcnemar_exact_p']:.4f}")
    print(f"  plots: {len(written)} files in {out / 'plots'}")
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
