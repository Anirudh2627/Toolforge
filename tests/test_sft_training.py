"""End-to-end SFT training tests on a tiny random-weight model (CPU).

Skipped when the training stack is not installed — the core suite stays
dependency-free. These tests exercise the real code path used on Colab:
settings loading, the transformers-version shims, LoRA wrapping, the masked
collator, Trainer stepping, checkpointing/resume, and adapter save/load.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from toolforge.checkpointing import find_latest_checkpoint, library_versions  # noqa: E402
from toolforge.config import ConfigError  # noqa: E402
from toolforge.sft import (  # noqa: E402
    SftSettings,
    _ta_kwargs,
    load_sft_settings,
    make_collator,
    resolve_precision,
)
from toolforge.sft_data import IGNORE_INDEX  # noqa: E402

from test_sft_data import SAMPLE_MESSAGES, TinyTokenizer  # noqa: E402
from toolforge.sft_data import tokenize_sft_messages  # noqa: E402


# --------------------------------------------------------------------------
# Settings (no torch required for these, but the module import already gated)
# --------------------------------------------------------------------------

class TestSettings:
    def test_load_repo_config(self):
        from pathlib import Path
        repo_config = Path(__file__).resolve().parents[1] / "configs" / "sft.yaml"
        settings = load_sft_settings(repo_config)
        assert settings.model_id == "Qwen/Qwen3-0.6B"
        assert settings.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj")

    def test_overrides_and_strict_keys(self, tmp_path):
        cfg = tmp_path / "sft.yaml"
        cfg.write_text("model_id: m\nlearning_rate: 1e-5\nlora:\n  r: 4\n", encoding="utf-8")
        s = load_sft_settings(cfg, overrides={"model_id": "other"})
        assert s.model_id == "other" and s.learning_rate == 1e-5 and s.lora.r == 4
        bad = tmp_path / "bad.yaml"
        bad.write_text("modle_id: typo\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="modle_id"):
            load_sft_settings(bad)

    def test_ta_kwargs_valid_for_installed_transformers(self, tmp_path):
        from transformers import TrainingArguments
        settings = dataclasses.replace(SftSettings(output_dir=str(tmp_path)), precision="fp32")
        kwargs = _ta_kwargs(settings, fp16=False, bf16=False, use_cpu=True)
        args = TrainingArguments(**kwargs)  # must not raise on this version
        assert args.per_device_train_batch_size == settings.per_device_batch_size

    def test_precision_resolution_cpu(self):
        assert resolve_precision(SftSettings(precision="auto"), "cpu") == "fp32"
        assert resolve_precision(SftSettings(precision="bf16"), "cpu") == "bf16"


# --------------------------------------------------------------------------
# Tiny-model training
# --------------------------------------------------------------------------

def build_tiny_model(tokenizer: TinyTokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=len(tokenizer.vocab) + 8, hidden_size=32,
                      intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return LlamaForCausalLM(cfg)


def build_dataset(tokenizer: TinyTokenizer, variants: int = 8):
    tok = tokenize_sft_messages(SAMPLE_MESSAGES, tokenizer)
    samples = [tok] * max(1, variants)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, idx):
            return samples[idx]

    return DS()


class TestTinyTraining:
    @pytest.fixture
    def tokenizer(self):
        return TinyTokenizer(extra_words=(
            "you are toolforge create event sync tomorrow calling calendar now "
            "ok true created done the").split())

    def test_two_step_training_and_checkpoint(self, tokenizer, tmp_path):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Trainer, TrainingArguments, set_seed

        set_seed(7)
        model = get_peft_model(build_tiny_model(tokenizer),
                               LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                          task_type=TaskType.CAUSAL_LM))
        args = TrainingArguments(
            output_dir=str(tmp_path / "trainer"), per_device_train_batch_size=2,
            gradient_accumulation_steps=1, max_steps=2, learning_rate=1e-3,
            logging_steps=1, eval_strategy="no", save_strategy="steps", save_steps=2,
            save_total_limit=2, report_to=[], remove_unused_columns=False,
            use_cpu=True, seed=7, optim="adamw_torch")
        trainer = Trainer(model=model, args=args, train_dataset=build_dataset(tokenizer),
                          data_collator=make_collator(tokenizer.pad_token_id))
        outcome = trainer.train()
        assert outcome.training_loss > 0
        ckpt = find_latest_checkpoint(args.output_dir)
        assert ckpt is not None and ckpt.name == "checkpoint-2"

    def test_loss_decreases_on_repeated_sample(self, tokenizer, tmp_path):
        """Gradients must flow to assistant spans: a tiny model memorizes one
        repeated conversation within a few dozen steps."""
        from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed

        set_seed(11)
        model = build_tiny_model(tokenizer)
        losses: list[float] = []

        class LogGrab(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses.append(float(logs["loss"]))

        args = TrainingArguments(
            output_dir=str(tmp_path / "mem"), per_device_train_batch_size=2,
            max_steps=40, learning_rate=3e-2, logging_steps=2, eval_strategy="no",
            save_strategy="no", report_to=[], remove_unused_columns=False,
            use_cpu=True, seed=11, optim="adamw_torch", warmup_steps=0)
        trainer = Trainer(model=model, args=args, train_dataset=build_dataset(tokenizer, 2),
                          data_collator=make_collator(tokenizer.pad_token_id),
                          callbacks=[LogGrab()])
        trainer.train()
        assert len(losses) >= 10
        assert losses[-1] < losses[0] * 0.6, f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    def test_adapter_save_and_reload(self, tokenizer, tmp_path):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        base = build_tiny_model(tokenizer)
        model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"],
                                                task_type=TaskType.CAUSAL_LM))
        adapter_dir = tmp_path / "adapter"
        model.save_pretrained(str(adapter_dir))
        assert (adapter_dir / "adapter_config.json").exists()

        reloaded = PeftModel.from_pretrained(build_tiny_model(tokenizer), str(adapter_dir))
        batch = make_collator(tokenizer.pad_token_id)(
            [tokenize_sft_messages(SAMPLE_MESSAGES, tokenizer)])
        out = reloaded(**batch)
        assert torch.isfinite(out.loss)
        # Labels are masked everywhere except assistant spans.
        supervised = (batch["labels"] != IGNORE_INDEX).sum().item()
        assert 0 < supervised < batch["labels"].numel()

    def test_resume_from_checkpoint(self, tokenizer, tmp_path):
        from transformers import Trainer, TrainingArguments, set_seed

        set_seed(13)

        def make_trainer():
            model = build_tiny_model(tokenizer)
            args = TrainingArguments(
                output_dir=str(tmp_path / "resume"), per_device_train_batch_size=2,
                max_steps=4, learning_rate=1e-3, logging_steps=1, eval_strategy="no",
                save_strategy="steps", save_steps=2, save_total_limit=2, report_to=[],
                remove_unused_columns=False, use_cpu=True, seed=13, optim="adamw_torch")
            return Trainer(model=model, args=args, train_dataset=build_dataset(tokenizer, 4),
                           data_collator=make_collator(tokenizer.pad_token_id)), args

        trainer, args = make_trainer()
        trainer.train()  # produces checkpoint-2 and checkpoint-4
        ckpt = find_latest_checkpoint(args.output_dir)
        assert ckpt is not None and ckpt.name == "checkpoint-4"

        trainer2, _ = make_trainer()
        outcome = trainer2.train(resume_from_checkpoint=str(ckpt))
        assert outcome.global_step == 4  # resumed at the end; no extra steps needed

    def test_library_versions_recorded(self):
        versions = library_versions()
        assert versions["torch"] != "not-installed"
        assert versions["transformers"] == transformers.__version__
        assert versions["peft"] == peft.__version__
