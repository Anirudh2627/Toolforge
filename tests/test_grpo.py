"""GRPO tests: advantage math, sample construction, optimization mechanics,
full train loop with checkpoints/resume, and the divergence detector.

Model-dependent tests run on a tiny random-weight PEFT model (CPU, offline);
they skip cleanly when torch/transformers/peft are not installed.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from toolforge.executor import EpisodeResult, ScriptedPolicy
from toolforge.grpo import (
    GrpoError,
    GrpoSettings,
    GrpoTrainer,
    episode_training_messages,
    group_advantages,
    load_grpo_settings,
)
from toolforge.rewards import RewardConfig, RewardConfigError
from toolforge.sft_data import IGNORE_INDEX, tokenize_sft_messages
from toolforge.tasks import save_tasks
from toolforge.verifier import verify_episode

from conftest import make_calendar_task

_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="training stack not installed")

CREATE = {"action": "tool_call", "tool": "calendar.create_event",
          "arguments": {"title": "Sync", "date": "2026-03-04", "start": "10:00", "end": "11:00"}}
FINAL_OK = {"action": "final_answer", "content": "Created the Sync event."}
FINAL_LIE = {"action": "final_answer", "content": "Done!"}
CLARIFY = {"action": "clarify", "question": "Which day works for you?"}


# --------------------------------------------------------------------------
# Pure-math tests (no torch)
# --------------------------------------------------------------------------

class TestGroupAdvantages:
    def test_centered_and_scaled(self):
        adv = group_advantages([1.0, 0.0, 0.5])
        assert abs(sum(adv)) < 1e-9                    # centered
        assert max(adv) > 0 > min(adv)
        # unit-ish scale: population std of advantages ~ 1
        mean = sum(adv) / len(adv)
        std = (sum((a - mean) ** 2 for a in adv) / len(adv)) ** 0.5
        assert abs(std - 1.0) < 1e-3

    def test_scale_false_centers_only(self):
        adv = group_advantages([1.0, 0.0], scale=False)
        assert adv == [0.5, -0.5]

    def test_zero_variance_group(self):
        assert group_advantages([0.7, 0.7, 0.7]) == [0.0, 0.0, 0.0] or \
               all(abs(a) < 1e-6 for a in group_advantages([0.7, 0.7, 0.7]))

    def test_requires_group(self):
        with pytest.raises(GrpoError):
            group_advantages([1.0])


class TestSettings:
    def test_validation(self):
        with pytest.raises(GrpoError, match="num_generations"):
            GrpoSettings(num_generations=1).validate()
        with pytest.raises(GrpoError, match="test split"):
            GrpoSettings(split="test").validate()
        with pytest.raises(GrpoError, match="clip_eps"):
            GrpoSettings(clip_eps=0.9).validate()

    def test_reward_dominance_gate(self):
        with pytest.raises(RewardConfigError):
            GrpoSettings(reward=RewardConfig(success_reward=0.01,
                                             invalid_call_penalty=-0.05)).validate()

    def test_yaml_loading_strict(self, tmp_path):
        cfg = tmp_path / "grpo.yaml"
        cfg.write_text("num_generations: 4\nlearning_rate: 1e-6\n"
                       "reward:\n  success_reward: 1.0\nlora:\n  r: 8\n", encoding="utf-8")
        s = load_grpo_settings(cfg)
        assert s.num_generations == 4 and s.learning_rate == 1e-6 and s.lora.r == 8
        bad = tmp_path / "bad.yaml"
        bad.write_text("num_generationz: 4\n", encoding="utf-8")
        from toolforge.config import ConfigError
        with pytest.raises(ConfigError, match="num_generationz"):
            load_grpo_settings(bad)

    def test_repo_config_loads(self):
        from pathlib import Path
        repo_cfg = Path(__file__).resolve().parents[1] / "configs" / "grpo.yaml"
        s = load_grpo_settings(repo_cfg)
        assert s.num_generations == 6 and s.beta_kl == 0.02 and s.reward.format_bonus == 0.0


class TestEpisodeMessages:
    def test_final_answer_episode_untouched(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        ep = executor.run(task, ScriptedPolicy([CREATE, FINAL_OK]))
        msgs = episode_training_messages(task, ep, compact_catalog=False)
        assert msgs[-1]["role"] == "assistant"
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"

    def test_step_limit_trailing_tool_message_trimmed(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture, max_steps=2)
        ep = executor.run(task, ScriptedPolicy([CREATE, CREATE]))  # never terminates
        assert ep.termination == "step_limit"
        msgs = episode_training_messages(task, ep, compact_catalog=False)
        assert msgs[-1]["role"] == "assistant"  # trailing tool observation removed
        assert json.loads(msgs[-1]["content"])["action"] == "tool_call"

    def test_policy_error_episodes_excluded(self):
        ep = EpisodeResult(task_id="t", termination="policy_error")
        task = make_calendar_task()
        assert episode_training_messages(task, ep, compact_catalog=False) is None

    def test_compact_catalog_applied(self, executor, small_calendar_fixture):
        task = make_calendar_task(fixture=small_calendar_fixture)
        ep = executor.run(task, ScriptedPolicy([FINAL_OK]))
        full = episode_training_messages(task, ep, compact_catalog=False)
        compact = episode_training_messages(task, ep, compact_catalog=True)
        assert len(compact[0]["content"]) < len(full[0]["content"])
        assert "- calendar.create_event(" in compact[0]["content"]
        assert compact[1:] == full[1:]


class TestDivergenceDetector:
    def _trainer(self):
        return GrpoTrainer(GrpoSettings(divergence_threshold=0.05))

    def test_flags_reward_up_accuracy_down(self):
        history = [
            {"iteration": 4, "window_reward_mean": 0.20, "eval": {"success_rate": 0.50}},
            {"iteration": 9, "window_reward_mean": 0.45, "eval": {"success_rate": 0.30}},
        ]
        assert self._trainer().check_divergence(history) is True

    def test_no_flag_when_accuracy_holds(self):
        history = [
            {"iteration": 4, "window_reward_mean": 0.20, "eval": {"success_rate": 0.50}},
            {"iteration": 9, "window_reward_mean": 0.45, "eval": {"success_rate": 0.48}},
        ]
        assert self._trainer().check_divergence(history) is False

    def test_no_flag_when_reward_falls(self):
        history = [
            {"iteration": 4, "window_reward_mean": 0.50, "eval": {"success_rate": 0.50}},
            {"iteration": 9, "window_reward_mean": 0.30, "eval": {"success_rate": 0.30}},
        ]
        assert self._trainer().check_divergence(history) is False

    def test_needs_two_evals(self):
        assert self._trainer().check_divergence(
            [{"iteration": 1, "window_reward_mean": 0.2, "eval": {"success_rate": 0.1}}]) is False


# --------------------------------------------------------------------------
# Tiny-model tests (torch stack)
# --------------------------------------------------------------------------

def _tiny_stack(vocab_words):
    """Tiny PEFT-wrapped Llama + whitespace tokenizer (offline)."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from test_sft_data import TinyTokenizer
    from test_sft_training import build_tiny_model

    tok = TinyTokenizer(extra_words=vocab_words)
    base = build_tiny_model(tok)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                                            target_modules=["q_proj", "v_proj"],
                                            task_type=TaskType.CAUSAL_LM))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-2, weight_decay=0.0)
    return model, tok, optimizer


VOCAB = ("you are toolforge create event sync tomorrow calling calendar now ok true "
         "created done the action tool_call final_answer title date start end "
         "2026-03-04 10:00 11:00 Sync Created").split()


def _injected_trainer(settings: GrpoSettings):
    model, tok, optimizer = _tiny_stack(VOCAB)
    trainer = GrpoTrainer(settings)
    trainer._model = model
    trainer._tokenizer = tok
    trainer._optimizer = optimizer
    trainer._model_info = {"device": "cpu", "precision": "fp32", "trainable_params": 1}
    return trainer, model, tok


class _RequestKeyedPolicy:
    """Scripted policy keyed by request substring; alternates success/lie per group
    member so every group has reward variance."""

    policy_id = "test-keyed"

    def __init__(self, mode: str):
        self.mode = mode          # "success" | "lie" | "clarify"
        self._index = 0

    def __call__(self, ctx) -> str:
        if ctx.step == 0:
            self._index = 0
        if "ambiguous" in ctx.request.lower() or "sometime" in ctx.request.lower():
            return json.dumps(CLARIFY)
        if self.mode == "lie" and self._index == 0:
            self._index += 1
            return json.dumps(FINAL_LIE)
        self._index += 1
        if self._index == 1:
            return json.dumps(CREATE)
        return json.dumps(FINAL_OK)


@requires_torch
class TestBuildSamples:
    def test_samples_from_real_episodes(self, executor, small_calendar_fixture):
        settings = GrpoSettings(num_generations=2, beta_kl=0.0, max_seq_len=2048,
                                compact_catalog=True)
        trainer, model, tok = _injected_trainer(settings)
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        good = executor.run(task, ScriptedPolicy([CREATE, FINAL_OK]))
        bad = executor.run(task, ScriptedPolicy([FINAL_LIE]))
        samples, stats = trainer.build_samples(task, [good, bad],
                                               [verify_episode(task, good),
                                                verify_episode(task, bad)],
                                               [1.0, 0.0])
        assert stats.kept == 2 and stats.successes == 1
        adv = [s.advantage for s in samples]
        assert adv[0] > 0 > adv[1]
        for s in samples:
            assert len(s.old_logprobs) == len(s.supervised_positions)
            assert all(p > 0 for p in s.supervised_positions)
            # supervised positions match the SFT masking of the same messages
            # under the SAME catalog rendering the trainer used (compact).
            msgs = episode_training_messages(task, good if s is samples[0] else bad,
                                             compact_catalog=True)
            ref = tokenize_sft_messages(msgs, tok, max_seq_len=2048,
                                        chat_template_kwargs={"enable_thinking": False})
            expected_positions = [i for i, l in enumerate(ref["labels"])
                                  if l != IGNORE_INDEX and i > 0]
            assert s.supervised_positions == expected_positions

    def test_zero_variance_group_skipped(self, executor, small_calendar_fixture):
        settings = GrpoSettings(num_generations=2, beta_kl=0.0)
        trainer, _, _ = _injected_trainer(settings)
        task = make_calendar_task(fixture=small_calendar_fixture)
        eps = [executor.run(task, ScriptedPolicy([CREATE, FINAL_OK])) for _ in range(2)]
        vs = [verify_episode(task, e) for e in eps]
        samples, stats = trainer.build_samples(task, eps, vs, [1.0, 1.0])
        assert samples == [] and stats.dropped_zero_variance == 2

    def test_too_long_dropped(self, executor, small_calendar_fixture):
        settings = GrpoSettings(num_generations=2, beta_kl=0.0, max_seq_len=5)
        trainer, _, _ = _injected_trainer(settings)
        task = make_calendar_task(fixture=small_calendar_fixture)
        ep = executor.run(task, ScriptedPolicy([CREATE, FINAL_OK]))
        samples, stats = trainer.build_samples(task, [ep, ep],
                                               [verify_episode(task, ep)] * 2, [1.0, 0.0])
        assert samples == [] and stats.dropped_too_long == 2


@requires_torch
class TestOptimization:
    def _samples(self, executor, small_calendar_fixture, trainer):
        task = make_calendar_task(fixture=small_calendar_fixture,
                                  checks=[{"check": "record_exists", "collection": "events",
                                           "where": {"title": "Sync"}}])
        good = executor.run(task, ScriptedPolicy([CREATE, FINAL_OK]))
        bad = executor.run(task, ScriptedPolicy([FINAL_LIE]))
        samples, _ = trainer.build_samples(task, [good, bad],
                                           [verify_episode(task, good),
                                            verify_episode(task, bad)], [1.0, 0.0])
        return samples

    def test_step_updates_only_trainable_params(self, executor, small_calendar_fixture):
        import torch
        settings = GrpoSettings(num_generations=2, beta_kl=0.0, ppo_epochs=1,
                                max_seq_len=2048, compact_catalog=True)
        trainer, model, _ = _injected_trainer(settings)
        samples = self._samples(executor, small_calendar_fixture, trainer)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        frozen_before = next(p.detach().clone() for p in model.parameters() if not p.requires_grad)
        metrics = trainer.optimization_pass(samples)
        assert metrics["steps"] == 2 and metrics["loss"] is not None
        changed = any(not torch.equal(before[n], p.detach())
                      for n, p in model.named_parameters() if p.requires_grad)
        assert changed, "trainable parameters must move"
        frozen_after = next(p.detach() for p in model.parameters() if not p.requires_grad)
        assert torch.equal(frozen_before, frozen_after), "base weights must stay frozen"

    def test_zero_advantage_zero_gradient(self, executor, small_calendar_fixture):
        import torch
        settings = GrpoSettings(num_generations=2, beta_kl=0.0, max_seq_len=2048,
                                compact_catalog=True)
        trainer, model, _ = _injected_trainer(settings)
        samples = self._samples(executor, small_calendar_fixture, trainer)
        for s in samples:
            s.advantage = 0.0
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        trainer.optimization_pass(samples)
        assert all(torch.equal(before[n], p.detach())
                   for n, p in model.named_parameters() if p.requires_grad)

    def test_kl_path_runs_with_adapter_disable(self, executor, small_calendar_fixture):
        settings = GrpoSettings(num_generations=2, beta_kl=0.05, max_seq_len=2048,
                                compact_catalog=True)
        trainer, model, _ = _injected_trainer(settings)
        samples = self._samples(executor, small_calendar_fixture, trainer)
        metrics = trainer.optimization_pass(samples)
        # LoRA B matrices init to zero -> policy == reference at start -> KL ~ 0
        assert metrics["kl"] >= 0.0 and metrics["kl"] < 1e-3
        assert metrics["steps"] == 2


@requires_torch
class TestFullLoop:
    def _write_tasks(self, tmp_path, small_calendar_fixture):
        create_task = make_calendar_task(
            task_id="tf-grpo-create", request="Create the Sync event on 2026-03-04.",
            fixture=small_calendar_fixture,
            checks=[{"check": "record_exists", "collection": "events",
                     "where": {"title": "Sync"}}])
        create_task.split = "train"
        clarify_task = make_calendar_task(
            task_id="tf-grpo-clarify", request="Schedule a meeting sometime next week.",
            fixture=small_calendar_fixture, terminal="clarify", forbid=True)
        clarify_task.split = "train"
        eval_task = make_calendar_task(
            task_id="tf-grpo-eval", request="Create the Sync event on 2026-03-04.",
            fixture=small_calendar_fixture,
            checks=[{"check": "record_exists", "collection": "events",
                     "where": {"title": "Sync"}}])
        eval_task.split = "val"
        path = tmp_path / "tasks.jsonl"
        save_tasks(path, [create_task, clarify_task, eval_task])
        return path

    def test_train_eval_checkpoint_resume(self, tmp_path, small_calendar_fixture):
        tasks_file = self._write_tasks(tmp_path, small_calendar_fixture)
        out = tmp_path / "grpo-out"
        settings = GrpoSettings(
            tasks_file=str(tasks_file), output_dir=str(out), num_generations=2,
            rollout_batch_size=1, total_iterations=2, eval_every=2, save_every=2,
            warmup_steps=0, beta_kl=0.0, max_seq_len=2048, compact_catalog=True,
            learning_rate=1e-2,
            task_limit=2, eval_task_limit=1, seed=5)
        trainer, model, tok = _injected_trainer(settings)

        calls = {"n": 0}

        def rollout_factory():
            calls["n"] += 1
            return _RequestKeyedPolicy("success" if calls["n"] % 2 else "lie")

        trainer._policy_factory_override = rollout_factory
        trainer._eval_policy_factory_override = lambda: _RequestKeyedPolicy("success")

        summary = trainer.train()
        assert summary["iterations_run"] == 2
        assert summary["total_rollouts"] == 4
        assert summary["total_successes"] >= 2          # alternating success policy
        history = [json.loads(l) for l in
                   (out / "history.jsonl").read_text().splitlines() if l.strip()]
        assert len(history) == 2
        assert history[1]["eval"]["success_rate"] == 1.0  # scripted eval policy succeeds
        ckpt = out / "checkpoint-iter00001"
        assert (ckpt / "adapter_config.json").exists()
        assert (ckpt / "training_state.pt").exists()
        assert (out / "run_metadata.json").exists()

        # Resume: one more iteration continues from the checkpoint.
        settings2 = GrpoSettings(**{**settings.__dict__, "total_iterations": 3,
                                    "resume": True})
        trainer2, _, _ = _injected_trainer(settings2)
        calls2 = {"n": 0}

        def rollout_factory2():
            calls2["n"] += 1
            return _RequestKeyedPolicy("success" if calls2["n"] % 2 else "lie")

        trainer2._policy_factory_override = rollout_factory2
        trainer2._eval_policy_factory_override = lambda: _RequestKeyedPolicy("success")
        summary2 = trainer2.train()
        history2 = [json.loads(l) for l in
                    (out / "history.jsonl").read_text().splitlines() if l.strip()]
        assert len(history2) == 3
        assert summary2["iterations_run"] == 3

    def test_refuses_test_split(self, tmp_path):
        settings = GrpoSettings(split="test")
        with pytest.raises(GrpoError):
            GrpoTrainer(settings)
