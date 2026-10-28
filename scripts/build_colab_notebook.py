#!/usr/bin/env python3
"""Build notebooks/ToolForge_Colab.ipynb (regenerable, valid nbformat 4.5).

The notebook is generated from this script so it stays reproducible and
reviewable as code. Run:  python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "notebooks" / "ToolForge_Colab.ipynb"

CELLS: list[tuple[str, str]] = []  # (cell_type, source)


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


# ---------------------------------------------------------------------------

md("""
# ToolForge — Colab pipeline (SFT + GRPO + verified evaluation)

**Read first.** This notebook drives the whole project end-to-end on a free Colab T4.
Ground rules it follows:

- ⚠️ **GPU TIME** markers flag every cell that uses the GPU; everything else is CPU-only.
- **Smoke vs full**: set `RUN_MODE` in Section 0. Smoke configurations are tiny and exist
  to validate the stack cheaply before you spend GPU hours.
- **No reruns of expensive stages**: `run_if_missing()` skips completed work based on
  output markers; delete a marker (or set `FORCE = True`) to redo a stage.
- **Sessions die.** Checkpoints, predictions, and history are written under `OUT_DIR`
  (optionally Google Drive). Training and GRPO support `--resume`. Re-run the notebook
  top-to-bottom after a disconnect; completed stages skip themselves.
- **No secrets by default.** The only credential anywhere is the optional HF token in
  Section 17, entered via `getpass` (never echoed, never written to disk).
- **No fabricated results.** Every number this notebook shows comes from a run you just
  executed or from saved artifacts on disk.

Requirements: a Colab GPU runtime (T4 or better) for ⚠️ sections; ~10 GB disk for the
0.6B model plus checkpoints (more for 1.7B).
""")

md("""
## 0. Configuration (edit this cell, then Run All or step through)

Note: markers do not distinguish smoke from full runs. If you ran smoke first and now
switch to `RUN_MODE="full"`, set `FORCE = True` (or delete the relevant marker
directories) for the stages you want re-executed at full budget.
""")

code('''import os
from pathlib import Path

# ---- edit these -----------------------------------------------------------
RUN_MODE   = "smoke"        # "smoke" (minutes, tiny budgets) | "full" (real experiment)
USE_DRIVE  = False          # True -> persist OUT_DIR on Google Drive across sessions
REPO_URL   = os.environ.get("TOOLFORGE_REPO_URL", "")   # your git remote, or "" + zip upload
REPO_DIR   = Path(os.environ.get("TOOLFORGE_REPO", "/content/ToolForge"))
MODEL_DEBUG = "Qwen/Qwen3-0.6B"    # pipeline debugging
MODEL_MAIN  = "Qwen/Qwen3-1.7B"    # main experiment (1.7B-class; needs T4 + QLoRA settings)
FORCE      = False                 # True -> ignore run_if_missing markers
# ---------------------------------------------------------------------------

MODEL_ID = MODEL_DEBUG if RUN_MODE == "smoke" else MODEL_MAIN
EVAL_LIMIT = 8 if RUN_MODE == "smoke" else None      # None -> whole split
SFT_EPOCHS = 1 if RUN_MODE == "smoke" else 3
GRPO_ITERS = 3 if RUN_MODE == "smoke" else 60

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = Path(os.environ.get("TOOLFORGE_OUT", "/content/drive/MyDrive/toolforge"))
else:
    OUT_DIR = Path(os.environ.get("TOOLFORGE_OUT", "/content/toolforge-outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR, CKPT_DIR = OUT_DIR / "eval", OUT_DIR / "checkpoints"
EVAL_DIR.mkdir(exist_ok=True); CKPT_DIR.mkdir(exist_ok=True)

import subprocess, sys, time

def sh(cmd: str, cwd: Path = REPO_DIR, check: bool = True) -> int:
    """Run a shell command with live output; fail loudly by default."""
    print(f"+ {cmd}")
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True, cwd=str(cwd)).returncode
    print(f"  ({time.time() - t0:.1f}s, exit {rc})")
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}")
    return rc

def run_if_missing(marker: Path, cmd: str) -> None:
    if marker.exists() and not FORCE:
        print(f"[skip] marker exists: {marker}  (set FORCE=True or delete it to re-run)")
    else:
        sh(cmd)

def show_metrics(run_dir: Path) -> None:
    m = json.loads((run_dir / "metrics.json").read_text())
    ci = m.get("wilson_ci") or {}
    print(f"{run_dir.name}: success {m['success_rate']:.3f} "
          f"(wilson95 [{ci.get('low', 0):.3f}, {ci.get('high', 0):.3f}]) on {m['n_tasks']} tasks")

import json
print(f"RUN_MODE={RUN_MODE}  MODEL_ID={MODEL_ID}  OUT_DIR={OUT_DIR}")''')

md("## 1. Check GPU, Python, disk")

code('''!nvidia-smi || echo "NO GPU — attach a GPU runtime (⚠️ sections need one)"
import sys, shutil
print("python:", sys.version.split()[0])
total, used, free = shutil.disk_usage("/")
print(f"disk: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
try:
    import torch
    print("preinstalled torch:", torch.__version__, "| cuda:", torch.version.cuda,
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except ImportError:
    print("torch not installed yet (Section 3 installs the training stack)")''')

md("""
## 2. Get the repository

Either clone from your remote (`REPO_URL` in Section 0) or upload a zip of the repo via
the Colab file panel and unzip it to `/content/ToolForge`. If `OUT_DIR` is on Drive and
already contains a repo copy, you can also point `REPO_DIR` there.
""")

code('''if REPO_URL:
    if not REPO_DIR.exists():
        sh(f"git clone {REPO_URL} {REPO_DIR}", cwd=Path("/content"))
    else:
        sh("git pull", cwd=REPO_DIR)
elif not (REPO_DIR / "pyproject.toml").exists():
    raise SystemExit(
        "No REPO_URL set and no repo at REPO_DIR.\\n"
        "Upload ToolForge.zip via the file panel, then run:\\n"
        "  !unzip -q /content/ToolForge.zip -d /content")
sh("ls", cwd=REPO_DIR)''')

md("""
## 3. Install pinned dependencies

Colab ships CUDA torch — we keep it and install the rest from `requirements-train.txt`
(pinned set verified together; see `scripts/check_train_env.py`). The compatibility probe
at the end must pass **before** any GPU work.
""")

code('''sh("python -c \\"import torch; print(torch.__version__)\\"")
sh("pip install -q -r requirements-train.txt")
sh("pip install -q -e '.[dev,report]'")
sh("pip install -q -r requirements-demo.txt || true")  # demo deps are optional here''')

code('''# Compatibility probe (network needed once for the tokenizer probe).
# The bitsandbytes probe FAILS on CPU-only boxes by design; on a GPU runtime all
# probes must pass before continuing.
sh("python scripts/check_train_env.py --model-id " + MODEL_ID)''')

md("## 4. Environment unit tests (CPU — no GPU, no downloads)")

code('''sh("python -m pytest tests/ -q")   # ~440 tests, <30s; must be green before spending GPU time''')

md("## 5. Generate benchmark tasks (CPU, deterministic)")

code('''run_if_missing(REPO_DIR / "data/tasks/benchmark_v1.jsonl",
               "python scripts/generate_tasks.py --config configs/base.yaml")
sh("head -c 600 data/tasks/benchmark_v1.jsonl")
print()
import json as _json
print(_json.dumps(_json.loads((REPO_DIR / "data/tasks/benchmark_v1_summary.json").read_text())["by_split"], indent=2))''')

md("""
## 6. Base-model smoke test ⚠️ GPU TIME

Downloads the base model once (~1.2 GB for 0.6B) and runs a tiny verified evaluation so
you see the whole loop working before committing to training. Smoke: 8 test tasks.
""")

code('''_limit = f"--limit {EVAL_LIMIT}" if EVAL_LIMIT else ""
run_if_missing(EVAL_DIR / "base-smoke" / "metrics.json",
               f"python scripts/evaluate.py --policy hf --model-id {MODEL_ID} "
               f"--split test {_limit} --run-name base-smoke --out '{EVAL_DIR}'")
show_metrics(EVAL_DIR / "base-smoke")''')

md("""
## 7. Collect trajectories

Reference solutions are **compiled from the benchmark's own specs, executed, and
verified** — CPU-only and deterministic (133/133 on train+val). Optional self-rollouts
(⚠️ GPU) demonstrate the rejection-sampling path and its (usually poor) acceptance rate
for an untuned model — that is expected and is exactly why reference data seeds SFT.
""")

code('''run_if_missing(REPO_DIR / "data/trajectories/reference_trainval.jsonl",
               "python scripts/generate_trajectories.py --source reference")
sh("cat data/trajectories/reference_trainval_stats.json | head -c 600")''')

code('''# OPTIONAL ⚠️ GPU TIME — self-generated rollouts with rejection sampling.
# Uncomment to run; then inspect data/trajectories/rollouts_self_hf_stats.json
# for the acceptance (rejection) rate — an untuned base model is expected to
# verify very few rollouts, which is exactly why reference data seeds SFT.
#
# sh(f"python scripts/generate_trajectories.py --source rollout --policy hf "
#    f"--model-id {MODEL_ID} --samples-per-task 4 --do-sample")''')

md("## 8. Prepare + validate SFT data (CPU)")

code('''run_if_missing(REPO_DIR / "data/sft/sft_train.jsonl",
               "python scripts/prepare_sft_data.py "
               "--trajectories data/trajectories/reference_trainval.jsonl --catalog compact")
sh("cat data/sft/sft_stats.json")
# Serialization self-check against the REAL tokenizer (masking correctness):
sh(f"python scripts/validate_qwen3_serialization.py {MODEL_ID}")''')

md("""
## 9. Run SFT ⚠️ GPU TIME (the main cost of the project)

Smoke: 2 optimizer steps on ≤8 samples (stack check, ~2 min). Full: 3 epochs on the whole
train set — on a T4 with 0.6B + QLoRA settings expect roughly 10–30 min (measure your own;
never trust anyone's guess, including this one). The run is **resumable**: if the session
dies, re-run this cell — `--resume` continues from the latest checkpoint.
""")

code('''SFT_OUT = CKPT_DIR / "sft"
_smoke = " --smoke" if RUN_MODE == "smoke" else ""
run_if_missing(SFT_OUT / "adapter-final" / "adapter_config.json",
               f"python scripts/train_sft.py --config configs/sft.yaml"
               f" --model-id {MODEL_ID} --output-dir '{SFT_OUT}'{_smoke}"
               f" --use-4bit --gradient-checkpointing --resume"
               + ("" if _smoke else f" --epochs {SFT_EPOCHS}"))''')

md("## 10. Evaluate the SFT checkpoint ⚠️ GPU TIME (val first, then test)")

code('''_limit = f"--limit {EVAL_LIMIT}" if EVAL_LIMIT else ""
ADAPTER = SFT_OUT / "adapter-final"
run_if_missing(EVAL_DIR / "sft-val" / "metrics.json",
               f"python scripts/evaluate.py --policy hf --model-id {MODEL_ID} "
               f"--adapter-id '{ADAPTER}' --split val --compact-catalog "
               f"--run-name sft-val --out '{EVAL_DIR}'")
show_metrics(EVAL_DIR / "sft-val")''')

md("""
## 11. GRPO smoke test ⚠️ GPU TIME

Two iterations with G=2 on 2 tasks — verifies the full RL loop (rollouts through the
executor, verifier rewards, group advantages, clipped update, checkpoint) cheaply.
""")

code('''GRPO_OUT = CKPT_DIR / "grpo"
assert ADAPTER and Path(ADAPTER).exists(), "Run Sections 9-10 (SFT) before GRPO"
if RUN_MODE == "smoke":
    run_if_missing(GRPO_OUT / "history.jsonl",
                   f"python scripts/train_grpo.py --config configs/grpo.yaml --smoke "
                   f"--model-id {MODEL_ID} --adapter-id '{ADAPTER}' --output-dir '{GRPO_OUT}' "
                   f"--compact-catalog --tasks-file {REPO_DIR}/data/tasks/benchmark_v1.jsonl")
    sh(f"tail -n 2 {GRPO_OUT}/history.jsonl || true")
else:
    print("Full GRPO runs in Section 12.")''')

md("""
## 12. Full GRPO experiment ⚠️ GPU TIME (budget carefully)

Each iteration runs `rollout_batch_size × num_generations` multi-turn episodes —
generation-bound on a T4. After the first iterations, read `seconds` per iteration from
`history.jsonl` and decide whether `total_iterations` fits your session budget. The run
is resumable at iteration boundaries; re-run this cell after a disconnect with
`--resume` (already included).
""")

code('''if RUN_MODE == "full":
    run_if_missing(GRPO_OUT / "run_metadata.json",
                   f"python scripts/train_grpo.py --config configs/grpo.yaml "
                   f"--model-id {MODEL_ID} --adapter-id '{ADAPTER}' --output-dir '{GRPO_OUT}' "
                   f"--compact-catalog --resume --iterations {GRPO_ITERS} "
                   f"--tasks-file {REPO_DIR}/data/tasks/benchmark_v1.jsonl")
    sh(f"tail -n 3 {GRPO_OUT}/history.jsonl")
else:
    print("Smoke mode: full GRPO skipped (set RUN_MODE='full' in Section 0).")''')

md("""
## 13. Evaluate all checkpoints on the test split ⚠️ GPU TIME

Same tasks, tools, step limits, verifier for every model. All three runs use
`--compact-catalog` so the serialization is identical across models (the SFT/GRPO
checkpoints were trained on compact catalogs; the base model is evaluated with the same
rendering for comparability — recorded in each run's `run_config.json`).
""")

code('''_limit = f"--limit {EVAL_LIMIT}" if EVAL_LIMIT else ""
GRPO_CKPT = sorted((GRPO_OUT).glob("checkpoint-iter*"))
GRPO_CKPT = GRPO_CKPT[-1] if GRPO_CKPT else ADAPTER  # fall back to SFT if GRPO absent

run_if_missing(EVAL_DIR / "test-base" / "metrics.json",
               f"python scripts/evaluate.py --policy hf --model-id {MODEL_ID} --split test "
               f"{_limit} --compact-catalog --run-name test-base --out {EVAL_DIR}")
run_if_missing(EVAL_DIR / "test-sft" / "metrics.json",
               f"python scripts/evaluate.py --policy hf --model-id {MODEL_ID} "
               f"--adapter-id '{ADAPTER}' --split test {_limit} --compact-catalog "
               f"--run-name test-sft --out '{EVAL_DIR}'")
run_if_missing(EVAL_DIR / "test-grpo" / "metrics.json",
               f"python scripts/evaluate.py --policy hf --model-id {MODEL_ID} "
               f"--adapter-id '{GRPO_CKPT}' --split test {_limit} --compact-catalog "
               f"--run-name test-grpo --out '{EVAL_DIR}'")
for name in ("test-base", "test-sft", "test-grpo"):
    if (EVAL_DIR / name / "metrics.json").exists():
        show_metrics(EVAL_DIR / name)''')

md("""
## 14. Ablation plots + report (CPU)

`compare_models.py` hard-checks comparability (grammar, tool fingerprints, split) and
re-verifies every `metrics.json` against its `predictions.jsonl` before comparing.
Missing runs are simply not compared — nothing is imputed.
""")

code('''_runs = []
for label, name in (("base", "test-base"), ("sft", "test-sft"), ("sft-grpo", "test-grpo")):
    if (EVAL_DIR / name / "metrics.json").exists():
        _runs.append(f"--run {label}='{EVAL_DIR / name}'")
if len(_runs) >= 2:
    _hist = f"--history sft-grpo='{GRPO_OUT}/history.jsonl'" if (GRPO_OUT / "history.jsonl").exists() else ""
    _abl = OUT_DIR / "ablations"
    sh("python scripts/compare_models.py " + " ".join(_runs) + f" {_hist} --out '{_abl}'")
    from IPython.display import Markdown, Image, display
    display(Markdown((OUT_DIR / "ablations" / "REPORT.md").read_text()[:4000]))
    for png in sorted((OUT_DIR / "ablations" / "plots").glob("*.png")):
        display(Image(filename=str(png)))
else:
    print("Need at least two evaluated checkpoints (Section 13) before comparing.")''')

md("""
## 15. Benchmark inference ⚠️ GPU TIME

Measures load time, warm-up, TTFT (streaming), per-turn/per-episode latency (median/p95),
throughput, peak GPU memory, and tokens. **Cost per 1,000 tasks is only computed if you
pass an hourly rate you actually pay/assume** — the output is then labeled as an estimate
with its basis. No invented prices, ever.
""")

code('''_bout = OUT_DIR / "benchmark.json"
_n_bench = 4 if RUN_MODE == "smoke" else 12
sh(f"python scripts/benchmark_inference.py --policy hf --model-id {MODEL_ID} "
   f"--adapter-id '{GRPO_CKPT}' --n-tasks {_n_bench} --ttft --out '{_bout}'")
# To get a labeled cost estimate, append e.g.:
#   --gpu-cost-per-hour 0.35 --hardware-label "rented T4 (vast.ai)"
import json as _j
_bench = OUT_DIR / "benchmark.json"
_b = _j.loads(_bench.read_text()) if _bench.exists() else None
print("peak GPU bytes:", (_b or {}).get("gpu_peak_memory_bytes"))''')

md("""
## 16. Export adapters + save artifacts

`export_model.py` merges the adapter into base weights (⚠️ ~1.2–3.5 GB disk; the merged
model is subject to the BASE model's license, not this repo's MIT license). With
`USE_DRIVE=True`, artifacts already live on Drive; otherwise copy what matters now —
the session filesystem is ephemeral.
""")

code('''# Optional merged export (skip on smoke to save disk/time):
if RUN_MODE == "full":
    _mout = OUT_DIR / "merged-sft"
    run_if_missing(_mout / "config.json",
                   f"python scripts/export_model.py --adapter '{ADAPTER}' "
                   f"--base-model {MODEL_ID} --out '{_mout}'")

if not USE_DRIVE:
    print("⚠️ USE_DRIVE=False: everything under OUT_DIR lives on the ephemeral Colab VM.\\n"
          "Download what matters (adapters, metrics, predictions, history) via the file "
          "panel,\\nor re-run Section 0 with USE_DRIVE=True before your next long stage.")
sh(f"du -sh '{OUT_DIR}'/* 2>/dev/null | sort -h | tail -12", cwd=Path("/content"))
print("\\nKeep: eval/*/metrics.json + predictions.jsonl, checkpoints/*/adapter-final, "
      "grpo history.jsonl, ablations/ — these are the reproducible record of the run.")''')

md("""
## 17. Publish to Hugging Face (optional, explicitly gated)

Off by default. If you enable it: the token is entered via `getpass` (never echoed,
never written to any file), and only **your** artifacts are uploaded (adapters, benchmark,
eval summaries). Review the base model's license before publishing adapters or merged
weights — Qwen3 models carry their own terms; this repo's MIT license covers code and
generated data only. Never upload Drive contents, credentials, or private data.
""")

code('''PUBLISH = False   # <-- set True only if you intend to publish
if PUBLISH:
    import getpass
    HF_USER = input("HF username: ").strip()
    HF_TOKEN = getpass.getpass("HF token (input hidden): ")
    sh("pip install -q huggingface_hub")
    from huggingface_hub import login, create_repo, upload_folder, upload_file
    login(token=HF_TOKEN, new_session=False)
    for name, folder in ((f"{HF_USER}-toolforge-sft-adapter", ADAPTER),
                         (f"{HF_USER}-toolforge-grpo-adapter", GRPO_CKPT)):
        create_repo(name, repo_type="model", exist_ok=True)
        upload_folder(folder_path=str(folder), repo_id=name,
                      ignore_patterns=["optimizer*", "rng_state*", "*.pt"])
        print("uploaded model:", name)
    create_repo(f"{HF_USER}-toolforge-benchmark", repo_type="dataset", exist_ok=True)
    upload_file(path_or_fileobj=str(REPO_DIR / "data/tasks/benchmark_v1.jsonl"),
                path_in_repo="benchmark_v1.jsonl",
                repo_id=f"{HF_USER}-toolforge-benchmark", repo_type="dataset")
    print("uploaded dataset: benchmark_v1.jsonl")
else:
    print("Publishing disabled (PUBLISH = False). Nothing was uploaded.")''')

md("""
---

### Session-dropped checklist

1. Reattach a GPU runtime; **Run All** — completed stages skip via markers.
2. If outputs were local (`USE_DRIVE=False`) and the VM was recycled, the runs are gone;
   re-run from Section 6 (model cache may survive within the same VM only).
3. Training/GRPO resume from `checkpoint-*` automatically (`--resume` is pre-set).

### What to send back when asking for help

- `outputs/eval/*/metrics.json` + the failing cell's full traceback,
- `scripts/check_train_env.py` output,
- `history.jsonl` tail for GRPO issues, and the `RUN_MODE`/model you used.
""")


def build_notebook() -> dict:
    cells = []
    for cell_type, source in CELLS:
        lines = [l + "\n" for l in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": cell_type, "id": uuid.uuid4().hex[:12], "metadata": {},
                "source": lines}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": "ToolForge_Colab.ipynb", "provenance": [], "toc_visible": True},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    OUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(nb['cells'])} cells)")
    try:
        import nbformat
        parsed = nbformat.read(str(OUT_PATH), as_version=4)
        nbformat.validate(parsed)
        print("nbformat validation: OK")
    except ImportError:
        json.loads(OUT_PATH.read_text())
        print("nbformat not installed; JSON parse check passed (install nbformat to validate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
