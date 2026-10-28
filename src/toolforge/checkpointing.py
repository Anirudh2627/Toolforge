"""Checkpoint and artifact utilities: fingerprints, metadata, resume, export.

Everything a training run persists goes through here so runs are auditable:
data fingerprints, library versions, seeds, and the exact settings used.
``export_merged_model`` produces a standalone model directory (adapter merged
into the base) for inference engines that cannot consume PEFT adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from toolforge import __version__, get_logger
from toolforge.schemas import ToolForgeError

logger = get_logger("checkpointing")

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class CheckpointError(ToolForgeError):
    """Raised for corrupt or unusable checkpoint/artifact states."""


def data_fingerprint(path: str | Path) -> str:
    """Stable content hash (first 16 hex chars of sha256) of a data file."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"Cannot fingerprint missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def save_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON via a temp file + rename so crashes never leave partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Highest-numbered ``checkpoint-N`` dir that actually contains weights."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    best: tuple[int, Path] | None = None
    for child in output_dir.iterdir():
        m = _CHECKPOINT_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        has_weights = any(child.glob("adapter_model.*")) or any(child.glob("model.safetensors*"))
        if has_weights and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), child)
    return best[1] if best else None


def library_versions() -> dict[str, str]:
    """Versions of every library that could affect reproducibility."""
    import sys
    versions = {"python": sys.version.split()[0], "toolforge": __version__}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets",
                 "bitsandbytes"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


def save_run_metadata(out_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist ``run_metadata.json`` (settings, seeds, versions, fingerprints)."""
    path = Path(out_dir) / "run_metadata.json"
    save_json_atomic(path, metadata)
    logger.info("Run metadata written to %s", path)
    return path


def export_merged_model(adapter_dir: str | Path, base_model_id: str, out_dir: str | Path,
                        dtype: str = "auto") -> Path:
    """Merge a PEFT adapter into its base model and save a standalone copy.

    Licensing note: the merged artifact contains base-model weights; its
    redistribution terms are the base model's, not this repo's MIT license.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise CheckpointError(f"Merging requires torch/transformers/peft: {exc}") from exc

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise CheckpointError(f"{adapter_dir} does not look like a PEFT adapter directory")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}.get(dtype, "auto")
    logger.info("Merging adapter %s into %s on %s", adapter_dir, base_model_id, device)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch_dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(str(out_dir))
    save_json_atomic(out_dir / "merge_info.json", {
        "base_model_id": base_model_id, "adapter_dir": str(adapter_dir),
        "adapter_fingerprint": data_fingerprint(adapter_dir / "adapter_config.json"),
        "versions": library_versions(), "dtype": dtype,
    })
    logger.info("Merged model saved to %s", out_dir)
    return out_dir
