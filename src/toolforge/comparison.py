"""Model/run comparison: load evaluation runs, verify comparability, analyze.

Integrity rules enforced here (the spec's "no fabricated comparisons"):

- Runs are only compared when they share grammar version, tool-catalog
  fingerprints, and evaluation split — mismatches raise unless the caller
  explicitly passes ``allow_incompatible=True`` (which is then recorded in the
  report, loudly).
- ``metrics.json`` is re-verified against ``predictions.jsonl`` (success rate
  and record count must match to 1e-9) so edited aggregate files cannot slip
  into a comparison.
- Paired analysis uses the task-id intersection and reports wins/losses/both/
  neither with an exact McNemar test (pure-python binomial; no scipy).
- CI vocabulary is explicit: per-run intervals are bootstrap-over-tasks /
  Wilson; run-to-run *training* variability exists only when repeated training
  runs are supplied via repeat-groups, and is labeled as such.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolforge.metrics import percentile
from toolforge.schemas import ToolForgeError


class ComparisonError(ToolForgeError):
    """Raised when runs cannot be compared or artifacts are inconsistent."""


@dataclass
class RunBundle:
    label: str
    path: Path
    metrics: dict[str, Any]
    run_config: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def split(self) -> str:
        return str(self.run_config.get("split", "?"))

    @property
    def model_description(self) -> str:
        return str(self.run_config.get("policy_description") or self.run_config.get("run_name"))


def load_run(label: str, path: str | Path, *, verify_metrics: bool = True) -> RunBundle:
    """Load one evaluation run directory (metrics.json, run_config.json, predictions.jsonl)."""
    run_dir = Path(path)
    missing = [name for name in ("metrics.json", "run_config.json", "predictions.jsonl")
               if not (run_dir / name).exists()]
    if missing:
        raise ComparisonError(f"Run {label!r} at {run_dir} is missing {missing}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    records = []
    with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ComparisonError(f"{run_dir}/predictions.jsonl:{line_no}: {exc}") from exc
    if not records:
        raise ComparisonError(f"Run {label!r}: predictions.jsonl contains no records")
    if verify_metrics:
        recomputed = sum(1 for r in records if r["success"]) / len(records)
        if abs(recomputed - metrics.get("success_rate", -1)) > 1e-9:
            raise ComparisonError(
                f"Run {label!r}: metrics.json success_rate ({metrics.get('success_rate')}) does "
                f"not match predictions.jsonl ({recomputed:.6f}); refusing to compare "
                "tampered or stale aggregates")
        if metrics.get("n_records") != len(records):
            raise ComparisonError(
                f"Run {label!r}: metrics.json n_records ({metrics.get('n_records')}) != "
                f"predictions ({len(records)})")
    return RunBundle(label=label, path=run_dir, metrics=metrics,
                     run_config=run_config, records=records)


def check_compatibility(bundles: list[RunBundle], *,
                        allow_incompatible: bool = False) -> list[str]:
    """Hard-check comparability; returns warnings (only with allow_incompatible)."""
    if len(bundles) < 2:
        raise ComparisonError("Comparison needs at least two runs")
    problems: list[str] = []

    grammars = {b.run_config.get("grammar_version") for b in bundles}
    if len(grammars) > 1:
        problems.append(f"grammar_version differs across runs: {sorted(str(g) for g in grammars)}")

    splits = {b.split for b in bundles}
    if len(splits) > 1:
        problems.append(f"evaluation split differs across runs: {sorted(splits)}")

    reference = bundles[0].run_config.get("tools_fingerprints") or {}
    for b in bundles[1:]:
        fps = b.run_config.get("tools_fingerprints") or {}
        for family_set, fp in reference.items():
            other = fps.get(family_set)
            if other is not None and other != fp:
                problems.append(
                    f"tool-catalog fingerprint mismatch for [{family_set}]: "
                    f"{bundles[0].label}={fp} vs {b.label}={other}")

    task_sets = [ {r["task_id"] for r in b.records} for b in bundles ]
    common = set.intersection(*task_sets)
    if not common:
        problems.append("runs share no task ids")
    elif len(common) < max(len(s) for s in task_sets):
        problems.append(
            f"task sets only partially overlap ({len(common)} common of "
            f"{[len(s) for s in task_sets]} per run); paired analysis uses the intersection")

    if problems and not allow_incompatible:
        raise ComparisonError(
            "Runs are not comparable:\n  - " + "\n  - ".join(problems)
            + "\nFix the runs or pass allow_incompatible=True (recorded in the report).")
    return [f"INCOMPATIBILITY OVERRIDE: {p}" for p in problems] if allow_incompatible else []


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------

def _latency_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    policy = [r["policy_seconds"] for r in records]
    wall = [r["policy_seconds"] + r["env_seconds"] for r in records]
    success_wall = [r["policy_seconds"] + r["env_seconds"] for r in records if r["success"]]
    return {
        "policy_seconds_median": percentile(policy, 0.5),
        "policy_seconds_p95": percentile(policy, 0.95),
        "wall_seconds_median": percentile(wall, 0.5),
        "time_to_success_wall_median": percentile(success_wall, 0.5) if success_wall else None,
    }


def comparison_table(bundles: list[RunBundle]) -> dict[str, dict[str, Any]]:
    """Per-label row combining metrics.json values with latency percentiles."""
    table: dict[str, dict[str, Any]] = {}
    for b in bundles:
        m = b.metrics
        table[b.label] = {
            "model": b.model_description,
            "run_path": str(b.path),
            "split": b.split,
            "grammar_version": b.run_config.get("grammar_version"),
            "tools_fingerprints": b.run_config.get("tools_fingerprints"),
            "n_records": m["n_records"],
            "n_tasks": m["n_tasks"],
            "success_rate": m["success_rate"],
            "wilson_ci": m.get("wilson_ci"),
            "bootstrap_ci": m.get("bootstrap_ci"),
            "held_out": m.get("held_out_family_success"),
            "seen": m.get("seen_family_success"),
            "multi_step": m.get("multi_step_success"),
            "by_category": {k: v["success_rate"] for k, v in m.get("success_by_category", {}).items()},
            "by_family": {k: v["success_rate"] for k, v in m.get("success_by_family", {}).items()},
            "invalid_json_rate": m.get("invalid_json_rate"),
            "invalid_tool_call_rate": m.get("invalid_tool_call_rate"),
            "argument_validation_failure_rate": m.get("argument_validation_failure_rate"),
            "step_limit_violation_rate": m.get("step_limit_violation_rate"),
            "loop_guard_rate": m.get("loop_guard_rate"),
            "unwanted_side_effect_rate": m.get("unwanted_side_effect_rate"),
            "clarification": {k: m.get("clarification", {}).get(k)
                              for k in ("precision", "recall", "f1")},
            "avg_tool_calls_per_task": m.get("avg_tool_calls_per_task"),
            "avg_turns_per_task": m.get("avg_turns_per_task"),
            "failure_category_counts": m.get("failure_category_counts", {}),
            **_latency_rows(b.records),
        }
    return table


def comparison_to_csv(table: dict[str, dict[str, Any]]) -> str:
    keys = ["success_rate", "held_out_rate", "seen_rate", "invalid_json_rate",
            "invalid_tool_call_rate", "step_limit_violation_rate",
            "unwanted_side_effect_rate", "avg_tool_calls_per_task",
            "policy_seconds_median", "n_tasks"]
    lines = ["label," + ",".join(keys)]
    for label, row in table.items():
        values = [
            row["success_rate"],
            (row.get("held_out") or {}).get("success_rate"),
            (row.get("seen") or {}).get("success_rate"),
            row.get("invalid_json_rate"), row.get("invalid_tool_call_rate"),
            row.get("step_limit_violation_rate"), row.get("unwanted_side_effect_rate"),
            row.get("avg_tool_calls_per_task"), row.get("policy_seconds_median"),
            row.get("n_tasks"),
        ]
        lines.append(label + "," + ",".join("" if v is None else
                                            (f"{v:.6f}" if isinstance(v, float) else str(v))
                                            for v in values))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Paired analysis
# --------------------------------------------------------------------------

def mcnemar_exact(wins_a: int, wins_b: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _per_task_success(bundle: RunBundle) -> dict[str, float]:
    per_task: dict[str, list[float]] = {}
    for r in bundle.records:
        per_task.setdefault(r["task_id"], []).append(1.0 if r["success"] else 0.0)
    return {tid: sum(v) / len(v) for tid, v in per_task.items()}


def paired_analysis(a: RunBundle, b: RunBundle, *, threshold: float = 0.5) -> dict[str, Any]:
    """Task-level paired comparison on the intersection (mean success >= threshold = solved)."""
    pa, pb = _per_task_success(a), _per_task_success(b)
    common = sorted(set(pa) & set(pb))
    if not common:
        raise ComparisonError(f"No common tasks between {a.label} and {b.label}")
    both = [t for t in common if pa[t] >= threshold and pb[t] >= threshold]
    only_a = [t for t in common if pa[t] >= threshold > pb[t]]
    only_b = [t for t in common if pb[t] >= threshold > pa[t]]
    neither = [t for t in common if pa[t] < threshold and pb[t] < threshold]
    return {
        "a": a.label, "b": b.label, "n_common_tasks": len(common),
        "both_solved": len(both), "only_a_solved": len(only_a),
        "only_b_solved": len(only_b), "neither_solved": len(neither),
        "mcnemar_exact_p": mcnemar_exact(len(only_a), len(only_b)),
        "task_ids_only_a": only_a, "task_ids_only_b": only_b,
    }


def repeat_group_stats(label: str, bundles: list[RunBundle]) -> dict[str, Any]:
    """Run-to-run TRAINING variability — only meaningful for independently
    trained runs of the same configuration (different seeds). Labeled clearly
    to distinguish it from bootstrap-over-tasks CIs."""
    rates = [b.metrics["success_rate"] for b in bundles]
    if len(rates) < 2:
        raise ComparisonError(f"repeat-group {label!r} needs >= 2 runs")
    mean = sum(rates) / len(rates)
    std = math.sqrt(sum((r - mean) ** 2 for r in rates) / (len(rates) - 1))
    return {"group": label, "n_runs": len(rates), "success_rates": rates,
            "mean_success_rate": mean, "std_success_rate": std,
            "ci_type": "repeated_training_runs (NOT bootstrap over tasks)"}


# --------------------------------------------------------------------------
# RL history (reward vs validated accuracy)
# --------------------------------------------------------------------------

def load_history(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise ComparisonError(f"History file not found: {p}")
    history = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not history:
        raise ComparisonError(f"History file is empty: {p}")
    return history


def rl_curve_data(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract (iteration, window reward) series and validated eval points."""
    iterations = [h["iteration"] for h in history]
    rewards = [h.get("window_reward_mean") for h in history]
    eval_points = [(h["iteration"], h["eval"]["success_rate"])
                   for h in history if h.get("eval")]
    divergences = [h["iteration"] for h in history if h.get("divergence_warning")]
    return {"iterations": iterations, "window_reward_mean": rewards,
            "eval_points": eval_points, "divergence_iterations": divergences}


# --------------------------------------------------------------------------
# Report generation (numbers come ONLY from the loaded artifacts)
# --------------------------------------------------------------------------

def find_regressions(bundles: list[RunBundle], table: dict[str, dict[str, Any]],
                     threshold: float = 0.05) -> list[dict[str, Any]]:
    """Categories where a later-listed run scores below an earlier-listed one.

    Order convention: pass runs in pipeline order (base, sft, sft+grpo).
    Only reports drops larger than ``threshold`` — smaller gaps are within
    noise for these sample sizes and are not called regressions.
    """
    regressions = []
    for i in range(len(bundles) - 1):
        a, b = bundles[i], bundles[i + 1]
        for cat in sorted(set(table[a.label]["by_category"]) & set(table[b.label]["by_category"])):
            ra, rb = table[a.label]["by_category"][cat], table[b.label]["by_category"][cat]
            if ra is not None and rb is not None and (ra - rb) > threshold:
                regressions.append({"from": a.label, "to": b.label, "category": cat,
                                    "rate_before": ra, "rate_after": rb,
                                    "drop": round(ra - rb, 4)})
    return regressions


def report_markdown(bundles: list[RunBundle], table: dict[str, dict[str, Any]],
                    paired: list[dict[str, Any]], groups: list[dict[str, Any]],
                    histories: dict[str, dict[str, Any]], warnings: list[str],
                    regressions: list[dict[str, Any]]) -> str:
    """Assemble REPORT.md. Every number is read from loaded artifacts."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        "# ToolForge Ablation Report",
        "",
        f"_Generated {now} from saved evaluation artifacts only. No number in this",
        "report was produced by anything other than the runs listed below._",
        "",
        "## Runs compared",
        "",
        "| label | model | split | n tasks | grammar | success (wilson95) |",
        "|---|---|---|---|---|---|",
    ]
    for b in bundles:
        row = table[b.label]
        ci = row.get("wilson_ci") or {}
        ci_txt = (f"[{ci.get('low', 0):.3f}, {ci.get('high', 0):.3f}]"
                  if ci.get("low") is not None else "n/a")
        lines.append(f"| {b.label} | {row['model']} | {row['split']} | {row['n_tasks']} "
                     f"| {row['grammar_version']} | **{row['success_rate']:.3f}** {ci_txt} |")
    if warnings:
        lines += ["", "> **Comparability warnings:**"] + [f"> - {w}" for w in warnings]

    # Headline (honest about statistical indistinguishability)
    ranked = sorted(bundles, key=lambda b: table[b.label]["success_rate"], reverse=True)
    best, second = ranked[0], ranked[1]
    lines += ["", "## Headline", ""]
    overlap = _ci_overlap(table[best.label].get("wilson_ci"), table[second.label].get("wilson_ci"))
    lines.append(f"- Strongest run: **{best.label}** at "
                 f"{table[best.label]['success_rate']:.3f} success rate.")
    if overlap:
        lines.append(f"- The gap to `{second.label}` is **not statistically distinguishable** "
                     "(Wilson 95% intervals overlap) at these sample sizes.")
    else:
        lines.append(f"- The gap to `{second.label}` does not show Wilson-95% overlap; "
                     "still check paired analysis before claiming a win.")

    # Held-out generalization
    lines += ["", "## Seen vs held-out tool families", "",
              "| label | seen success | held-out success | gap |", "|---|---|---|---|"]
    for b in bundles:
        row = table[b.label]
        seen = (row.get("seen") or {}).get("success_rate")
        held = (row.get("held_out") or {}).get("success_rate")
        gap = (seen - held) if (seen is not None and held is not None) else None
        lines.append(f"| {b.label} | {_fmt(seen)} | {_fmt(held)} | {_fmt(gap)} |")

    # Behavior rates
    lines += ["", "## Behavior rates", "",
              "| label | invalid JSON | invalid tool call | arg validation | step limit "
              "| side effects | clarify P/R |", "|---|---|---|---|---|---|---|"]
    for b in bundles:
        row = table[b.label]
        cl = row.get("clarification") or {}
        lines.append(
            f"| {b.label} | {_fmt(row.get('invalid_json_rate'))} "
            f"| {_fmt(row.get('invalid_tool_call_rate'))} "
            f"| {_fmt(row.get('argument_validation_failure_rate'))} "
            f"| {_fmt(row.get('step_limit_violation_rate'))} "
            f"| {_fmt(row.get('unwanted_side_effect_rate'))} "
            f"| {_fmt(cl.get('precision'))}/{_fmt(cl.get('recall'))} |")

    # Latency
    lines += ["", "## Latency (measured in the evaluation runs)", "",
              "| label | policy s median | policy s p95 | wall s median | time-to-success wall median |",
              "|---|---|---|---|---|"]
    for b in bundles:
        row = table[b.label]
        lines.append(f"| {b.label} | {_fmt(row.get('policy_seconds_median'))} "
                     f"| {_fmt(row.get('policy_seconds_p95'))} | {_fmt(row.get('wall_seconds_median'))} "
                     f"| {_fmt(row.get('time_to_success_wall_median'))} |")

    # Paired
    if paired:
        lines += ["", "## Paired task analysis (intersection of task sets)", ""]
        for p in paired:
            lines.append(
                f"- `{p['a']}` vs `{p['b']}` on {p['n_common_tasks']} tasks: "
                f"both {p['both_solved']}, only-{p['a']} {p['only_a_solved']}, "
                f"only-{p['b']} {p['only_b_solved']}, neither {p['neither_solved']}; "
                f"exact McNemar p = {p['mcnemar_exact_p']:.4f}")

    # Repeat groups (training variability)
    if groups:
        lines += ["", "## Repeated-training variability", ""]
        for g in groups:
            lines.append(f"- `{g['group']}`: {g['n_runs']} runs, success rates "
                         f"{[round(r, 4) for r in g['success_rates']]}, mean "
                         f"{g['mean_success_rate']:.4f} ± {g['std_success_rate']:.4f} "
                         "(_repeated training runs; distinct from bootstrap-over-tasks CIs_)")
    else:
        lines += ["", "## Repeated-training variability", "",
                  "- Not available: each configuration was trained once. Run-to-run training",
                  "  variability is therefore NOT captured; per-run CIs are bootstrap over tasks only."]

    # RL dynamics
    if histories:
        lines += ["", "## RL dynamics (training reward vs validated accuracy)", ""]
        for label, curve in histories.items():
            evals = curve["eval_points"]
            lines.append(f"- `{label}`: {len(curve['iterations'])} iterations, "
                         f"{len(evals)} validated evals"
                         + (f", divergence warnings at iterations {curve['divergence_iterations']}"
                            if curve["divergence_iterations"] else ", no divergence warnings") + ".")
            if len(evals) >= 2:
                first, last = evals[0], evals[-1]
                lines.append(f"  - validated success: iter {first[0]} = {first[1]:.3f} -> "
                             f"iter {last[0]} = {last[1]:.3f}")
    else:
        lines += ["", "## RL dynamics", "",
                  "- Not available: no GRPO history files were supplied (--history label=path)."]

    # Regressions & failure modes
    lines += ["", "## Regressions (drops > threshold between consecutive runs)", ""]
    if regressions:
        for r in regressions:
            lines.append(f"- `{r['from']}` -> `{r['to']}`: **{r['category']}** "
                         f"{r['rate_before']:.3f} -> {r['rate_after']:.3f} "
                         f"(drop {r['drop']:.3f})")
    else:
        lines.append("- None beyond the configured threshold.")

    lines += ["", "## Top failure categories per run", ""]
    for b in bundles:
        counts = table[b.label].get("failure_category_counts") or {}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        lines.append(f"- `{b.label}`: " + (", ".join(f"{k} ({v})" for k, v in top) or "none"))

    # Limitations (auto, honest)
    lines += ["", "## Limitations", "",
              "- Sample sizes above are the whole story for statistical power; overlapping CIs",
              "  mean differences must not be over-interpreted.",
              "- Retrieval/refusal categories are verified lexically (computed ground-truth",
              "  containment); state-change categories are verified from environment state.",
              "- Latency numbers are from the evaluation machines recorded in each run's",
              "  run_config.json; hardware may differ between runs — check before comparing.",
              "- This report reflects ONLY the supplied runs; nothing is imputed."]
    return "\n".join(lines) + "\n"


def _ci_overlap(ci_a: dict | None, ci_b: dict | None) -> bool:
    if not ci_a or not ci_b or ci_a.get("low") is None or ci_b.get("low") is None:
        return True  # unknown -> conservatively report as indistinguishable
    return ci_a["low"] <= ci_b["high"] and ci_b["low"] <= ci_a["high"]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
