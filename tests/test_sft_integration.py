"""Full run_sft integration test: settings → load → LoRA → tokenize → train →
adapter + metadata, using a tiny locally-saved model (no network, CPU).

This exercises the exact code path used on Colab, including the transformers
version shims, so a green run here means `scripts/train_sft.py --smoke` will
work given a real model id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from toolforge.sft import SftSettings, load_sft_settings, run_sft
from toolforge.sft_data import save_sft_records

from test_sft_data import SAMPLE_MESSAGES

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|fim_prefix|>{{ m['role'] }}\n{{ m['content'] }}<|fim_suffix|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|fim_prefix|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> str:
    """Save a tiny Llama + fast tokenizer with a prefix-stable chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = ["[PAD]", "[EOS]", "[UNK]"]
    words = set()
    for m in SAMPLE_MESSAGES:
        words.update(m["content"].split())
    for m in SAMPLE_MESSAGES:
        words.add(m["role"])
    vocab.extend(sorted(words))
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]",
                                        eos_token="[EOS]")
    tokenizer.chat_template = CHAT_TEMPLATE

    cfg = LlamaConfig(vocab_size=len(vocab) + 8, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=0, eos_token_id=1)
    model = LlamaForCausalLM(cfg)

    d = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(str(d))
    tokenizer.save_pretrained(str(d))
    return str(d)


@pytest.fixture(scope="module")
def sft_files(tmp_path_factory) -> tuple[str, str]:
    d = tmp_path_factory.mktemp("sft-data")
    records = []
    for i in range(6):
        messages = json.loads(json.dumps(SAMPLE_MESSAGES))
        messages[1]["content"] = f"{messages[1]['content']} variant {i}"
        records.append({"task_id": f"t{i}", "trajectory_id": f"traj{i}", "source": "reference",
                        "split": "train", "family": "calendar",
                        "category": "single_step_selection", "messages": messages})
    save_sft_records(d / "train.jsonl", records)
    save_sft_records(d / "val.jsonl", records[:2])
    return str(d / "train.jsonl"), str(d / "val.jsonl")


class TestRunSftEndToEnd:
    def test_full_smoke_run(self, tiny_model_dir, sft_files, tmp_path):
        train_file, val_file = sft_files
        out = tmp_path / "sft-out"
        settings = SftSettings(
            model_id=tiny_model_dir, train_file=train_file, val_file=val_file,
            output_dir=str(out), max_seq_len=256, max_steps=2, per_device_batch_size=2,
            grad_accum_steps=1, learning_rate=1e-3, warmup_steps=0, logging_steps=1,
            eval_steps=2, save_steps=2, save_total_limit=2, device="cpu", precision="fp32",
            seed=3)
        result = run_sft(settings)

        assert result.global_step == 2
        assert result.train_loss > 0
        assert result.eval_loss is not None and result.eval_loss > 0
        adapter = Path(result.adapter_dir)
        assert (adapter / "adapter_config.json").exists()
        assert any(adapter.glob("adapter_model.*"))
        assert (adapter / "sft_metadata.json").exists()

        meta = json.loads((out / "run_metadata.json").read_text())
        assert meta["phase"] == "sft"
        assert meta["model_info"]["device"] == "cpu"
        assert meta["model_info"]["trainable_params"] > 0
        assert meta["dataset_stats"]["train_samples"] == 6
        assert "train_file" in meta["dataset_stats"]["fingerprints"]
        assert meta["versions"]["transformers"] == transformers.__version__

    def test_settings_roundtrip_through_yaml(self, tiny_model_dir, sft_files, tmp_path):
        """load_sft_settings + overrides must produce the same run shape."""
        train_file, val_file = sft_files
        cfg = tmp_path / "sft.yaml"
        cfg.write_text(
            f"model_id: {tiny_model_dir}\ntrain_file: {train_file}\nval_file: {val_file}\n"
            f"output_dir: {tmp_path / 'yaml-out'}\nmax_steps: 1\nlearning_rate: 1e-3\n"
            "device: cpu\nprecision: fp32\ngrad_accum_steps: 1\nwarmup_steps: 0\n"
            "per_device_batch_size: 2\neval_steps: 1\nsave_steps: 1\nmax_seq_len: 256\n"
            "lora:\n  r: 2\n  target_modules: [q_proj, v_proj]\n",
            encoding="utf-8")
        settings = load_sft_settings(cfg)
        assert settings.lora.target_modules == ("q_proj", "v_proj")
        result = run_sft(settings)
        assert result.global_step == 1
