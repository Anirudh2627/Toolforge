#!/usr/bin/env python3
"""Compatibility probe: run this ON THE TRAINING MACHINE before any training.

Verifies the installed stack against what ToolForge's training code actually
calls — versions, renamed arguments, Qwen3 support, quantization availability,
and (for Phase 4) the TRL GRPO API shape. Prints a PASS/FAIL line per probe
and exits non-zero on any FAIL, so notebooks can gate on it.

Usage: python scripts/check_train_env.py [--model-id Qwen/Qwen3-0.6B]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS: list[tuple[str, bool, str]] = []


def probe(name: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
        except Exception as exc:  # any probe failure is a real incompatibility
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn
    return wrap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    @probe("python >= 3.10")
    def _py():
        assert sys.version_info >= (3, 10), sys.version
        return sys.version.split()[0]

    @probe("torch importable")
    def _torch():
        import torch
        return torch.__version__

    @probe("torch version >= 2.8")
    def _torch_ver():
        import torch
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        assert (major, minor) >= (2, 8), torch.__version__
        return torch.__version__

    @probe("CUDA availability (informational)")
    def _cuda():
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            return f"{name}, capability {cap[0]}.{cap[1]}, " \
                   f"{'bf16 ok' if cap[0] >= 8 else 'use fp16 (no fast bf16)'}"
        return "no CUDA — CPU mode (debugging only; 4-bit unavailable)"

    @probe("transformers >= 4.51 (Qwen3 support)")
    def _tf_ver():
        import transformers
        parts = transformers.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert (major, minor) >= (4, 51), transformers.__version__
        return transformers.__version__

    @probe("transformers Qwen3 architecture present")
    def _qwen3():
        import transformers.models  # noqa: F401
        from transformers.models import qwen3  # noqa: F401
        return "qwen3 module ok"

    @probe("TrainingArguments uses eval_strategy/evaluation_strategy consistently")
    def _ta():
        from transformers import TrainingArguments
        fields_ = TrainingArguments.__dataclass_fields__
        key = "eval_strategy" if "eval_strategy" in fields_ else "evaluation_strategy"
        assert key in fields_
        return f"resolved kwarg: {key}"

    @probe("model dtype kwarg resolution")
    def _dtype():
        from toolforge.sft import _model_dtype_kwarg_name
        return f"from_pretrained kwarg: {_model_dtype_kwarg_name()}"

    @probe("peft LoraConfig + TaskType")
    def _peft():
        import peft
        from peft import LoraConfig, TaskType, get_peft_model  # noqa: F401
        return peft.__version__

    @probe("bitsandbytes 4-bit (needs CUDA; FAIL on CPU is expected)")
    def _bnb():
        import torch
        import bitsandbytes
        assert torch.cuda.is_available(), "bitsandbytes NF4 requires CUDA"
        from transformers import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        return bitsandbytes.__version__

    @probe("accelerate importable")
    def _acc():
        import accelerate
        return accelerate.__version__

    @probe("trl GRPOTrainer API shape (Phase 4)")
    def _trl():
        import inspect
        import trl
        from trl import GRPOConfig, GRPOTrainer
        params = set(inspect.signature(GRPOTrainer.__init__).parameters)
        assert {"model", "reward_funcs", "args", "train_dataset"} <= params, sorted(params)
        fields_ = {f for f in GRPOConfig.__dataclass_fields__}
        assert {"num_generations", "max_completion_length"} <= fields_
        return f"trl {trl.__version__}: GRPOTrainer(reward_funcs=...) ok"

    @probe("tokenizer chat template loads (network needed once)")
    def _tok():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_id)
        msgs = [{"role": "user", "content": "hi"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert isinstance(text, str) and text
        return f"{args.model_id}: template renders ({len(text)} chars)"

    @probe("toolforge SFT serialization self-test (tiny fake tokenizer)")
    def _serialization():
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from test_sft_data import TinyTokenizer, SAMPLE_MESSAGES  # type: ignore
        from toolforge.sft_data import IGNORE_INDEX, tokenize_sft_messages
        tok = TinyTokenizer()
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        supervised = sum(1 for l in out["labels"] if l != IGNORE_INDEX)
        assert supervised > 0
        return f"{supervised}/{len(out['labels'])} tokens supervised"

    failed = 0
    print(f"{'PROBE':<62} STATUS  DETAIL")
    for name, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{name:<62} {status:<6}  {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} probes passed")
    if failed:
        print("Resolve FAILed probes before training (bitsandbytes-on-CPU is the "
              "one expected failure in CPU-only environments).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
