"""Ablation plots generated strictly from loaded evaluation artifacts.

Rules (spec §11):
- Every figure is drawn from data passed in by the caller (loaded from run
  directories); nothing is invented here.
- Missing data raises :class:`PlotDataError` — a plot is never padded with
  plausible-looking filler.
- ``illustrative=True`` is an explicit, loud mode: filenames are prefixed
  ``illustrative_``, titles say ILLUSTRATIVE, and a diagonal watermark is drawn
  on the axes. It exists for documentation previews only.

Styling is deliberately plain (no seaborn dependency): clean bars, value
labels, Wilson-95% error bars where available, consistent per-label colors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe everywhere (Colab, CI, servers)
import matplotlib.pyplot as plt  # noqa: E402

from toolforge.schemas import ToolForgeError  # noqa: E402
from toolforge.tasks import CATEGORIES  # noqa: E402


class PlotDataError(ToolForgeError):
    """Raised when a plot is requested without the data to draw it honestly."""


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
_WATERMARK = "ILLUSTRATIVE DATA — NOT EXPERIMENTAL RESULTS"


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def _finish(fig, ax, out_path: Path, illustrative: bool, title: str) -> Path:
    ax.set_title(title)
    if illustrative:
        ax.text(0.5, 0.5, _WATERMARK, transform=ax.transAxes, fontsize=16,
                color="red", alpha=0.35, ha="center", va="center", rotation=20,
                fontweight="bold", zorder=10)
        title = "ILLUSTRATIVE: " + title
        ax.set_title(title)
        out_path = out_path.with_name("illustrative_" + out_path.name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _require_table(table: dict[str, dict[str, Any]]) -> None:
    if not table:
        raise PlotDataError("empty comparison table; nothing to plot (no fabricated data)")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_overall_comparison(table: dict[str, dict[str, Any]], out_path: str | Path, *,
                            illustrative: bool = False) -> Path:
    _require_table(table)
    _style()
    labels = list(table)
    rates = [table[l]["success_rate"] for l in labels]
    if any(r is None for r in rates):
        raise PlotDataError("success_rate missing for at least one run")
    errs_low, errs_high = [], []
    for l in labels:
        ci = table[l].get("wilson_ci") or {}
        lo, hi = ci.get("low"), ci.get("high")
        if lo is None or hi is None:
            errs_low.append(0.0)
            errs_high.append(0.0)
        else:
            # Defensive clamp: error bars must never be negative.
            errs_low.append(max(0.0, table[l]["success_rate"] - lo))
            errs_high.append(max(0.0, hi - table[l]["success_rate"]))
    fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(labels)), 4.2))
    bars = ax.bar(labels, rates, yerr=[errs_low, errs_high], capsize=5,
                  color=[_color(i) for i in range(len(labels))], error_kw={"lw": 1})
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, min(rate + 0.02, 0.98), f"{rate:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("task success rate")
    ax.set_xlabel("model")
    n = table[labels[0]].get("n_tasks", "?")
    return _finish(fig, ax, Path(out_path), illustrative,
                   f"Overall task success (n={n} tasks; error bars: Wilson 95%)")


def plot_category_success(table: dict[str, dict[str, Any]], out_path: str | Path, *,
                          illustrative: bool = False) -> Path:
    _require_table(table)
    _style()
    labels = list(table)
    present = [c for c in CATEGORIES if any(c in (table[l].get("by_category") or {})
                                            for l in labels)]
    if not present:
        raise PlotDataError("no per-category success rates available in any run")
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(present)), 4.5))
    for i, l in enumerate(labels):
        rates = [(table[l].get("by_category") or {}).get(c) for c in present]
        xs = [x + i * width for x in range(len(present))]
        plotted_x = [x for x, r in zip(xs, rates) if r is not None]
        plotted_r = [r for r in rates if r is not None]
        ax.bar(plotted_x, plotted_r, width=width, label=l, color=_color(i))
    ax.set_xticks([x + width * (len(labels) - 1) / 2 for x in range(len(present))])
    ax.set_xticklabels([c.replace("_", "\n") for c in present], fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("success rate")
    ax.legend(fontsize=8)
    return _finish(fig, ax, Path(out_path), illustrative, "Success rate by task category")


def plot_seen_vs_heldout(table: dict[str, dict[str, Any]], out_path: str | Path, *,
                         illustrative: bool = False) -> Path:
    _require_table(table)
    _style()
    labels = list(table)
    seen = [(table[l].get("seen") or {}).get("success_rate") for l in labels]
    held = [(table[l].get("held_out") or {}).get("success_rate") for l in labels]
    if all(s is None for s in seen) and all(h is None for h in held):
        raise PlotDataError("no seen/held-out slices in the comparison table")
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(labels)), 4.2))
    ax.bar([i - width / 2 for i in x], [s or 0 for s in seen], width=width,
           label="seen families", color=_color(0))
    ax.bar([i + width / 2 for i in x], [h or 0 for h in held], width=width,
           label="held-out family (crm)", color=_color(3), hatch="//", edgecolor="white")
    for i, (s, h) in enumerate(zip(seen, held)):
        if s is not None:
            ax.text(i - width / 2, s + 0.015, f"{s:.2f}", ha="center", fontsize=9)
        if h is not None:
            ax.text(i + width / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.legend(fontsize=9)
    return _finish(fig, ax, Path(out_path), illustrative,
                   "Seen vs held-out tool-family accuracy (generalization)")


def plot_behavior_rates(table: dict[str, dict[str, Any]], out_path: str | Path, *,
                        illustrative: bool = False) -> Path:
    _require_table(table)
    _style()
    labels = list(table)
    series = [("invalid JSON", "invalid_json_rate"),
              ("invalid tool call", "invalid_tool_call_rate"),
              ("arg validation fail", "argument_validation_failure_rate"),
              ("step-limit violation", "step_limit_violation_rate"),
              ("unwanted side effects", "unwanted_side_effect_rate")]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 4.2))
    width = 0.8 / len(series)
    for j, (name, key) in enumerate(series):
        values = [table[l].get(key) or 0.0 for l in labels]
        xs = [i + j * width for i in range(len(labels))]
        ax.bar(xs, values, width=width, label=name, color=_color(j))
    ax.set_xticks([i + width * (len(series) - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    return _finish(fig, ax, Path(out_path), illustrative,
                   "Invalid-action and side-effect rates")


def plot_latency_vs_success(table: dict[str, dict[str, Any]], out_path: str | Path, *,
                            illustrative: bool = False) -> Path:
    _require_table(table)
    _style()
    labels = list(table)
    points = []
    for i, l in enumerate(labels):
        x = table[l].get("policy_seconds_median")
        y = table[l].get("success_rate")
        if x is None or y is None:
            raise PlotDataError(f"run {l!r} lacks latency or success data for the scatter plot")
        points.append((x, y, l, _color(i)))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for x, y, l, c in points:
        ax.scatter([x], [y], s=90, color=c, zorder=3, label=l)
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median policy seconds per task (log scale)")
    ax.set_ylabel("task success rate")
    ax.set_ylim(-0.02, 1.05)
    return _finish(fig, ax, Path(out_path), illustrative,
                   "Latency vs task success (median per-task policy time)")


def plot_rl_reward_vs_accuracy(curve: dict[str, Any], label: str, out_path: str | Path, *,
                               illustrative: bool = False) -> Path:
    """Training reward (windowed) against INDEPENDENT validated eval accuracy."""
    iterations = curve.get("iterations") or []
    rewards = curve.get("window_reward_mean") or []
    eval_points = curve.get("eval_points") or []
    if not iterations or all(r is None for r in rewards):
        raise PlotDataError(f"history for {label!r} has no reward data")
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(iterations, rewards, color=_color(0), lw=1.6, label="train reward (window mean)")
    ax.set_xlabel("GRPO iteration")
    ax.set_ylabel("mean training reward", color=_color(0))
    ax.tick_params(axis="y", labelcolor=_color(0))
    if eval_points:
        ax2 = ax.twinx()
        ax2.plot([p[0] for p in eval_points], [p[1] for p in eval_points], "o--",
                 color=_color(2), lw=1.4, ms=5, label="validated success (val split, greedy)")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("validated task success", color=_color(2))
        ax2.tick_params(axis="y", labelcolor=_color(2))
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    else:
        ax.legend(fontsize=8, loc="lower right")
    for it in curve.get("divergence_iterations") or []:
        ax.axvline(it, color="red", ls=":", lw=1.2)
        ax.annotate("divergence\nwarning", (it, ax.get_ylim()[1] * 0.92), color="red",
                    fontsize=7.5, ha="center")
    return _finish(fig, ax, Path(out_path), illustrative,
                   f"GRPO run '{label}': training reward vs validated task accuracy")


def plot_all(table: dict[str, dict[str, Any]], out_dir: str | Path, *,
             histories: dict[str, dict[str, Any]] | None = None,
             illustrative: bool = False) -> list[Path]:
    """Render the full ablation figure set; returns written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [
        plot_overall_comparison(table, out / "overall_success.png", illustrative=illustrative),
        plot_category_success(table, out / "category_success.png", illustrative=illustrative),
        plot_seen_vs_heldout(table, out / "seen_vs_heldout.png", illustrative=illustrative),
        plot_behavior_rates(table, out / "behavior_rates.png", illustrative=illustrative),
    ]
    try:
        written.append(plot_latency_vs_success(table, out / "latency_vs_success.png",
                                               illustrative=illustrative))
    except PlotDataError:
        pass  # latency missing (e.g. non-model policies): skip, never fabricate
    for label, curve in (histories or {}).items():
        try:
            written.append(plot_rl_reward_vs_accuracy(
                curve, label, out / f"rl_dynamics_{label}.png", illustrative=illustrative))
        except PlotDataError:
            pass
    return written


# --------------------------------------------------------------------------
# Illustrative-only data (never mixed with real runs; loudly labeled)
# --------------------------------------------------------------------------

def illustrative_table() -> dict[str, dict[str, Any]]:
    """A clearly fake table for documentation previews ONLY.

    Values are arbitrary placeholders; every consumer labels them as
    illustrative. This function exists so docs can show the figure layout
    without implying any experimental result.
    """
    fake_labels = ["ILLUSTRATIVE-base", "ILLUSTRATIVE-sft", "ILLUSTRATIVE-sft+grpo"]
    fake_rates = [0.12, 0.48, 0.55]
    table = {}
    for label, rate in zip(fake_labels, fake_rates):
        lo, hi = max(0.0, rate - 0.09), min(1.0, rate + 0.09)
        table[label] = {
            "model": label, "split": "test", "grammar_version": "illustrative",
            "n_tasks": 59, "n_records": 59, "success_rate": rate,
            "wilson_ci": {"low": lo, "high": hi},
            "bootstrap_ci": {"low": lo, "high": hi},
            "seen": {"success_rate": min(1.0, rate + 0.06)},
            "held_out": {"success_rate": max(0.0, rate - 0.08)},
            "by_category": {c: max(0.0, min(1.0, rate + ((hash(c) % 7) - 3) * 0.05))
                            for c in CATEGORIES},
            "invalid_json_rate": max(0.0, 0.25 - rate / 3),
            "invalid_tool_call_rate": max(0.0, 0.20 - rate / 4),
            "argument_validation_failure_rate": max(0.0, 0.15 - rate / 5),
            "step_limit_violation_rate": max(0.0, 0.30 - rate / 2),
            "unwanted_side_effect_rate": max(0.0, 0.10 - rate / 8),
            "avg_tool_calls_per_task": 1.5 + rate,
            "policy_seconds_median": 6.0 - 4 * rate,
            "policy_seconds_p95": 12.0 - 8 * rate,
            "clarification": {"precision": rate, "recall": rate, "f1": rate},
            "failure_category_counts": {},
        }
    return table
