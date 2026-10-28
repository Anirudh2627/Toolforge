"""QLoRA/LoRA supervised fine-tuning on verified ToolForge trajectories.

Design decisions (documented per the build spec):

- **Serialization** goes through :mod:`toolforge.sft_data`: the tokenizer's own
  chat template renders the executor's messages, loss is masked to assistant
  spans (including end-of-turn tokens), and over-long samples are dropped and
  counted — never silently truncated.
- **Trainer**: plain ``transformers.Trainer`` with a custom collator instead of
  TRL's SFTTrainer. Rationale: the mask semantics above are already exact, and
  avoiding TRL's version-drifting SFT API removes one compatibility risk. TRL
  is reserved for GRPO (Phase 4), where it is the actual algorithm provider.
- **Version compatibility**: transformers 4.5x and 5.x differ in argument
  names (``evaluation_strategy`` vs ``eval_strategy``, ``torch_dtype`` vs
  ``dtype``). Both are probed at runtime via a small shim instead of pinning
  users to one major version; ``scripts/check_train_env.py`` reports the
  resolved stack before training starts.
- **4-bit**: requested explicitly (``use_4bit``); if bitsandbytes is missing or
  no CUDA device exists, training FAILS rather than silently switching
  precision — a silent precision change would invalidate memory assumptions.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from toolforge import get_logger
from toolforge.checkpointing import (
    data_fingerprint,
    find_latest_checkpoint,
    library_versions,
    save_json_atomic,
    save_run_metadata,
)
from toolforge.config import ConfigError
from toolforge.sft_data import TokenizedSftDataset, tokenize_sft_file

logger = get_logger("sft")


class SftError(Exception):
    """Raised for training misconfiguration or missing dependencies."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    bias: str = "none"


@dataclass(frozen=True)
class SftSettings:
    model_id: str = "Qwen/Qwen3-0.6B"
    train_file: str = "data/sft/sft_train.jsonl"
    val_file: str = "data/sft/sft_val.jsonl"
    output_dir: str = "outputs/checkpoints/sft"
    max_seq_len: int = 1024
    num_epochs: float = 3.0
    max_steps: int = -1                     # >0 overrides num_epochs (smoke tests)
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    weight_decay: float = 0.0
    seed: int = 20260301
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 3
    use_4bit: bool = False
    gradient_checkpointing: bool = False
    precision: str = "auto"                 # auto|bf16|fp16|fp32
    optim: str = "adamw_torch"              # 4-bit runs typically: paged_adamw_8bit
    device: str = "auto"                    # auto|cpu|cuda
    resume: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    trust_remote_code: bool = False
    lora: LoraSettings = field(default_factory=LoraSettings)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)


def load_sft_settings(path: str | Path | None = None,
                      overrides: dict[str, Any] | None = None) -> SftSettings:
    """Load SFT settings from YAML with strict key validation."""
    raw: dict[str, Any] = {}
    if path is not None:
        file = Path(path)
        if not file.exists():
            raise ConfigError(f"SFT config not found: {file}")
        loaded = yaml.safe_load(file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError("SFT config root must be a mapping")
        raw = loaded
    if overrides:
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}

    lora_raw = raw.pop("lora", {})
    known_lora = {f.name for f in dataclasses.fields(LoraSettings)}
    if set(lora_raw) - known_lora:
        raise ConfigError(f"Unknown sft.lora keys: {sorted(set(lora_raw) - known_lora)}")
    target = lora_raw.get("target_modules")
    if isinstance(target, list):
        lora_raw["target_modules"] = tuple(target)
    _coerce_numeric_fields(LoraSettings, lora_raw, "sft.lora")
    lora = LoraSettings(**lora_raw)

    known = {f.name for f in dataclasses.fields(SftSettings)}
    if set(raw) - known:
        raise ConfigError(f"Unknown sft keys: {sorted(set(raw) - known)} (known: {sorted(known)})")
    _coerce_numeric_fields(SftSettings, raw, "sft")
    return SftSettings(lora=lora, **raw)


def _coerce_numeric_fields(cls: type, raw: dict[str, Any], where: str) -> None:
    """YAML 1.1 parses ``1e-5`` (no dot) as a STRING — a classic silent-typo trap.

    Coerce values whose dataclass annotation is int/float, failing loudly when
    the string is not a clean number. Anything else is left untouched so the
    dataclass constructor reports its own type errors.
    """
    import typing
    hints = typing.get_type_hints(cls)
    for name, value in list(raw.items()):
        hint = hints.get(name)
        if hint not in (int, float):
            continue
        if isinstance(value, bool):
            raise ConfigError(f"{where}.{name} must be a number, got bool")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise ConfigError(f"{where}.{name}={raw[name]!r} is not a number") from exc
        if isinstance(value, (int, float)):
            raw[name] = hint(value)
        else:
            raise ConfigError(f"{where}.{name} must be a number, got {type(value).__name__}")


# --------------------------------------------------------------------------
# Version shims (transformers 4.5x vs 5.x)
# --------------------------------------------------------------------------

def _training_arguments_cls() -> type:
    from transformers import TrainingArguments
    return TrainingArguments


def _ta_kwargs(settings: SftSettings, *, fp16: bool, bf16: bool, use_cpu: bool) -> dict[str, Any]:
    """TrainingArguments kwargs adapted to the installed transformers version."""
    from transformers import TrainingArguments
    fields_ = TrainingArguments.__dataclass_fields__
    eval_key = "eval_strategy" if "eval_strategy" in fields_ else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(settings.resolved_output_dir() / "trainer"),
        "per_device_train_batch_size": settings.per_device_batch_size,
        "per_device_eval_batch_size": max(1, settings.per_device_batch_size),
        "gradient_accumulation_steps": settings.grad_accum_steps,
        "learning_rate": settings.learning_rate,
        "num_train_epochs": settings.num_epochs,
        "max_steps": settings.max_steps,
        "warmup_steps": settings.warmup_steps,
        "weight_decay": settings.weight_decay,
        "logging_steps": settings.logging_steps,
        eval_key: "steps",
        "eval_steps": settings.eval_steps,
        "save_strategy": "steps",
        "save_steps": settings.save_steps,
        "save_total_limit": settings.save_total_limit,
        "seed": settings.seed,
        "data_seed": settings.seed,
        "fp16": fp16,
        "bf16": bf16,
        "optim": settings.optim,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "use_cpu": use_cpu,
    }
    if settings.max_steps > 0:  # smoke mode: keep eval/save cadence meaningful
        kwargs["num_train_epochs"] = max(settings.num_epochs, 1.0)
    return kwargs


def _model_dtype_kwarg_name() -> str:
    """transformers renamed ``torch_dtype`` -> ``dtype`` in 4.56; support both."""
    import transformers
    parts = transformers.__version__.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if major < 5 and len(parts) > 1 else 0
    return "dtype" if (major >= 5 or minor >= 56) else "torch_dtype"


def _from_pretrained_kwargs(dtype_name: str, quantization_config: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if quantization_config is not None:
        kw["quantization_config"] = quantization_config
    else:
        import torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype_name]
        kw[_model_dtype_kwarg_name()] = torch_dtype
    return kw


# --------------------------------------------------------------------------
# Model / data preparation
# --------------------------------------------------------------------------

def resolve_device(settings: SftSettings) -> str:
    if settings.device != "auto":
        return settings.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_precision(settings: SftSettings, device: str) -> str:
    if settings.precision != "auto":
        return settings.precision
    if device != "cuda":
        return "fp32"
    import torch
    major = torch.cuda.get_device_capability(0)[0]
    return "bf16" if major >= 8 else "fp16"  # T4 (7.5) -> fp16


def load_model_and_tokenizer(settings: SftSettings) -> tuple[Any, Any, dict[str, Any]]:
    """Load base model (+4-bit quantization if requested) and wrap with LoRA."""
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SftError(f"Training requires torch/transformers/peft (requirements-train.txt): {exc}") from exc

    device = resolve_device(settings)
    precision = resolve_precision(settings, device)
    quantization_config = None
    if settings.use_4bit:
        if device != "cuda":
            raise SftError("use_4bit=True requires CUDA (bitsandbytes NF4); "
                           "set use_4bit=False for CPU debugging runs")
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401  (fail early with a clear message)
        except ImportError as exc:
            raise SftError(f"use_4bit=True requires bitsandbytes: {exc}") from exc
        compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                         "fp32": torch.float32}[precision]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)

    logger.info("Loading %s (device=%s, precision=%s, 4bit=%s)",
                settings.model_id, device, precision, settings.use_4bit)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id,
                                              trust_remote_code=settings.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        trust_remote_code=settings.trust_remote_code,
        **_from_pretrained_kwargs(precision, quantization_config))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_seconds = time.perf_counter() - t0

    if settings.gradient_checkpointing:
        # Required so frozen-embedding gradients flow through checkpointed blocks.
        model.enable_input_require_grads()
        if not settings.use_4bit:
            model.gradient_checkpointing_enable()
    elif settings.use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=settings.lora.r, lora_alpha=settings.lora.alpha,
        lora_dropout=settings.lora.dropout, target_modules=list(settings.lora.target_modules),
        bias=settings.lora.bias, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA wrapped: %s trainable / %s total params (%.2f%%), load %.1fs",
                f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1), load_seconds)
    info = {"device": device, "precision": precision, "use_4bit": settings.use_4bit,
            "trainable_params": trainable, "total_params": total,
            "load_seconds": round(load_seconds, 2)}
    return model, tokenizer, info


class TorchSftDataset:
    """Minimal torch Dataset over tokenized samples (avoids a datasets dependency)."""

    def __init__(self, tokenized: TokenizedSftDataset) -> None:
        import torch  # noqa: F401  (fail early if torch missing)
        self._samples = tokenized.samples
        self.dropped_too_long = tokenized.dropped_too_long

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._samples[idx]


def make_collator(pad_token_id: int):
    """Right-pad input_ids/attention_mask; labels pad with -100 (no loss)."""
    import torch

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append(b["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention)}

    return collate


def prepare_datasets(settings: SftSettings, tokenizer: Any
                     ) -> tuple[TorchSftDataset, TorchSftDataset | None, dict[str, Any]]:
    train_tok = tokenize_sft_file(settings.train_file, tokenizer, settings.max_seq_len)
    if settings.max_train_samples:
        train_tok.samples = train_tok.samples[:settings.max_train_samples]
    val_tok = None
    val_path = Path(settings.val_file)
    if val_path.exists():
        val_tok = tokenize_sft_file(val_path, tokenizer, settings.max_seq_len)
        if settings.max_val_samples:
            val_tok.samples = val_tok.samples[:settings.max_val_samples]
    else:
        logger.warning("No validation file at %s; training without eval loss", val_path)
    stats = {
        "train_samples": len(train_tok), "train_dropped_too_long": train_tok.dropped_too_long,
        "val_samples": len(val_tok) if val_tok else 0,
        "val_dropped_too_long": val_tok.dropped_too_long if val_tok else 0,
        "train_file": str(settings.train_file), "val_file": str(settings.val_file),
    }
    return (TorchSftDataset(train_tok),
            TorchSftDataset(val_tok) if val_tok else None, stats)


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------

@dataclass
class TrainResult:
    output_dir: str
    adapter_dir: str
    train_loss: float
    eval_loss: float | None
    global_step: int
    dataset_stats: dict[str, Any]
    model_info: dict[str, Any]
    resumed_from: str | None
    metadata_path: str


def run_sft(settings: SftSettings) -> TrainResult:
    """Full SFT run: load, tokenize, train, evaluate, save adapter + metadata."""
    from transformers import Trainer, set_seed

    set_seed(settings.seed)
    model, tokenizer, model_info = load_model_and_tokenizer(settings)
    train_ds, val_ds, data_stats = prepare_datasets(settings, tokenizer)
    if len(train_ds) == 0:
        raise SftError(f"No training samples after tokenization ({settings.train_file}); "
                       "generate trajectories first (scripts/generate_trajectories.py)")

    device = model_info["device"]
    ta_kwargs = _ta_kwargs(settings, fp16=(model_info["precision"] == "fp16" and device == "cuda"),
                           bf16=(model_info["precision"] == "bf16" and device == "cuda"),
                           use_cpu=(device == "cpu"))
    if val_ds is None:
        ta_kwargs["eval_strategy" if "eval_strategy" in
                  _training_arguments_cls().__dataclass_fields__ else "evaluation_strategy"] = "no"
    args = _training_arguments_cls()(**ta_kwargs)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tokenizer.pad_token_id))

    resume_from = None
    if settings.resume:
        resume_from = find_latest_checkpoint(args.output_dir)
        if resume_from is None:
            logger.info("resume requested but no checkpoint under %s; starting fresh",
                        args.output_dir)
    logger.info("Starting SFT (resume=%s)", resume_from)
    train_outcome = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)

    eval_loss = None
    if val_ds is not None:
        eval_metrics = trainer.evaluate()
        eval_loss = float(eval_metrics["eval_loss"])
        logger.info("Final eval loss: %.4f", eval_loss)

    adapter_dir = settings.resolved_output_dir() / "adapter-final"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    fingerprints = {name: data_fingerprint(p) for name, p in
                    (("train_file", settings.train_file), ("val_file", settings.val_file))
                    if Path(p).exists()}
    metadata = {
        "phase": "sft",
        "settings": dataclasses.asdict(settings),
        "model_info": model_info,
        "dataset_stats": {**data_stats, "fingerprints": fingerprints},
        "train_loss": float(train_outcome.training_loss),
        "eval_loss": eval_loss,
        "global_step": int(train_outcome.global_step),
        "resumed_from": str(resume_from) if resume_from else None,
        "versions": library_versions(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = save_run_metadata(settings.resolved_output_dir(), metadata)
    save_json_atomic(adapter_dir / "sft_metadata.json", metadata)
    return TrainResult(output_dir=str(settings.resolved_output_dir()),
                       adapter_dir=str(adapter_dir),
                       train_loss=float(train_outcome.training_loss),
                       eval_loss=eval_loss, global_step=int(train_outcome.global_step),
                       dataset_stats=data_stats, model_info=model_info,
                       resumed_from=str(resume_from) if resume_from else None,
                       metadata_path=str(meta_path))
