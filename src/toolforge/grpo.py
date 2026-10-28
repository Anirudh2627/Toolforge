"""GRPO (Group Relative Policy Optimization) with verifier-backed rewards.

Why not TRL's GRPOTrainer (evidence-based; probed on trl==1.16.0, see
``scripts/check_train_env.py`` and the Phase 4 notes in README):

- TRL's multi-turn tool loop (``trl/trainer/grpo_trainer.py::_tool_call_loop``)
  dispatches **OpenAI-style native ``tool_calls``** — parsed by the chat-template
  harness — as stateless ``name(**arguments)`` Python calls. ToolForge's grammar
  is a single JSON action object inside message *content*
  (``tool_call``/``clarify``/``final_answer``); TRL's loop never detects it, so
  rollouts would never touch the environments and rewards would degenerate to
  text-only scoring.
- ToolForge rewards require per-rollout isolated environment execution,
  schema-validated error observations, loop guards, turn budgets, and final-state
  verification (:mod:`toolforge.verifier`) — inexpressible in TRL's
  ``reward_funcs(prompts, completions, ...)`` protocol.
- Adopting TRL would force the whole project (benchmark, SFT data, evaluation,
  demo) onto template-native tool calls, breaking the one-grammar contract.

Per the build spec ("implement a valid, tested alternative"), this module
implements the GRPO objective directly (DeepSeekMath formulation):

1. For each task prompt, sample ``G`` episodes from the current policy through
   the same :class:`~toolforge.executor.Executor` loop used for evaluation
   (environments run on CPU; the GPU only does generation/training).
2. Reward each episode with :func:`~toolforge.rewards.compute_reward` from the
   *verifier verdict* (actual final environment state — never textual claims).
3. Group-relative advantage: ``A_g = (r_g - mean(r)) / (std(r) + eps)``
   (division optional via ``scale_rewards``); zero-variance groups contribute
   no gradient and are skipped (logged).
4. Token-level clipped surrogate over **assistant tokens only** (identical
   masking to SFT, :mod:`toolforge.sft_data`):

   ``L = -E[min(rho_t A, clip(rho_t, 1-eps, 1+eps) A)] + beta * KL_k3(pi||pi_ref)``

   with ``rho_t = pi_theta / pi_old``. The reference policy is the base model
   with LoRA adapters disabled (``PeftModel.disable_adapter``) — no second
   model copy in memory.
5. ``ppo_epochs`` optimization passes per rollout batch; AdamW; linear warmup
   then constant LR; gradient clipping.

Everything is checkpointed at iteration boundaries (adapter + optimizer +
scheduler + RNG + history) because Colab sessions die; ``resume=True`` picks up
the latest checkpoint. Eval runs the *independent* verifier-based evaluation on
a fixed val-split subset with greedy decoding, and a divergence detector flags
"training reward up while validated task accuracy down" — the reward-hacking
early-warning the spec requires.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from toolforge import get_logger
from toolforge.checkpointing import data_fingerprint, library_versions, save_json_atomic
from toolforge.config import ConfigError
from toolforge.data_generation import compact_system_prompt
from toolforge.evaluation import preflight_tasks
from toolforge.executor import Executor
from toolforge.inference import HuggingFacePolicy
from toolforge.rewards import RewardConfig, compute_reward
from toolforge.registry import default_registry
from toolforge.schemas import GRAMMAR_VERSION
from toolforge.sft import (LoraSettings, _coerce_numeric_fields,
                           _model_dtype_kwarg_name, resolve_device, resolve_precision)
from toolforge.sft_data import IGNORE_INDEX, tokenize_sft_messages
from toolforge.tasks import Task, load_tasks, tasks_by_split
from toolforge.trajectories import messages_from_episode
from toolforge.verifier import verify_episode

logger = get_logger("grpo")


class GrpoError(Exception):
    """Raised for GRPO misconfiguration or unrecoverable training states."""


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GrpoSettings:
    model_id: str = "Qwen/Qwen3-0.6B"
    adapter_id: str | None = None            # REQUIRED in practice: the SFT checkpoint
    output_dir: str = "outputs/checkpoints/grpo"
    tasks_file: str = "data/tasks/benchmark_v1.jsonl"
    split: str = "train"
    task_limit: int | None = None
    # rollouts
    num_generations: int = 6                 # G (group size)
    rollout_batch_size: int = 2              # tasks per iteration
    max_new_tokens: int = 256                # per turn; actions are short JSON
    temperature: float = 1.0
    top_p: float = 0.95
    max_seq_len: int = 1536
    # objective
    ppo_epochs: int = 1                      # mu
    clip_eps: float = 0.2
    beta_kl: float = 0.02                    # 0 disables KL/ref forwards
    scale_rewards: bool = True
    skip_zero_variance: bool = True
    # optimization
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    total_iterations: int = 60               # outer GRPO iterations
    gradient_checkpointing: bool = False
    # schedule / eval / checkpoints
    eval_every: int = 5
    eval_split: str = "val"
    eval_task_limit: int = 12
    save_every: int = 5
    save_total_limit: int = 3
    log_every: int = 1
    divergence_threshold: float = 0.05       # accuracy drop that triggers a warning
    # system
    seed: int = 20260301
    device: str = "auto"
    dtype: str = "auto"                      # resolved like SFT (T4 -> fp16)
    use_4bit: bool = False                   # QLoRA+GRPO is experimental; off by default
    enable_thinking: bool = False
    compact_catalog: bool = False            # MUST match the SFT checkpoint's rendering
    trust_remote_code: bool = False
    resume: bool = False
    lora: LoraSettings = field(default_factory=LoraSettings)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def validate(self) -> None:
        if self.num_generations < 2:
            raise GrpoError("num_generations must be >= 2 (group-relative advantage needs a group)")
        if self.rollout_batch_size < 1:
            raise GrpoError("rollout_batch_size must be >= 1")
        if self.clip_eps <= 0 or self.clip_eps > 0.5:
            raise GrpoError(f"clip_eps out of sane range (0, 0.5]: {self.clip_eps}")
        if self.beta_kl < 0:
            raise GrpoError("beta_kl must be >= 0")
        if self.total_iterations < 1:
            raise GrpoError("total_iterations must be >= 1")
        if self.split == "test":
            raise GrpoError("GRPO must never train on the test split")
        self.reward.validate(max_steps=12)


def load_grpo_settings(path: str | Path | None = None,
                       overrides: dict[str, Any] | None = None) -> "GrpoSettings":
    """Load GRPO settings from YAML with strict key validation."""
    raw: dict[str, Any] = {}
    if path is not None:
        file = Path(path)
        if not file.exists():
            raise ConfigError(f"GRPO config not found: {file}")
        loaded = yaml.safe_load(file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError("GRPO config root must be a mapping")
        raw = dict(loaded)
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    lora_raw = dict(raw.pop("lora", {}))
    reward_raw = dict(raw.pop("reward", {}))
    known_lora = {f.name for f in dataclasses.fields(LoraSettings)}
    known_reward = {f.name for f in dataclasses.fields(RewardConfig)}
    if set(lora_raw) - known_lora:
        raise ConfigError(f"Unknown grpo.lora keys: {sorted(set(lora_raw) - known_lora)}")
    if set(reward_raw) - known_reward:
        raise ConfigError(f"Unknown grpo.reward keys: {sorted(set(reward_raw) - known_reward)}")
    if isinstance(lora_raw.get("target_modules"), list):
        lora_raw["target_modules"] = tuple(lora_raw["target_modules"])
    known = {f.name for f in dataclasses.fields(GrpoSettings)}
    if set(raw) - known:
        raise ConfigError(f"Unknown grpo keys: {sorted(set(raw) - known)} (known: {sorted(known)})")
    # YAML 1.1 parses '1e-6' (no dot) as a string — coerce like the SFT loader.
    _coerce_numeric_fields(GrpoSettings, raw, "grpo")
    _coerce_numeric_fields(LoraSettings, lora_raw, "grpo.lora")
    _coerce_numeric_fields(RewardConfig, reward_raw, "grpo.reward")
    settings = GrpoSettings(lora=LoraSettings(**lora_raw), reward=RewardConfig(**reward_raw), **raw)
    settings.validate()
    return settings


# --------------------------------------------------------------------------
# Advantage computation
# --------------------------------------------------------------------------

def group_advantages(rewards: list[float], *, scale: bool = True,
                     eps: float = 1e-6) -> list[float]:
    """GRPO group-relative advantages for one prompt's G rewards."""
    if len(rewards) < 2:
        raise GrpoError("group advantages require >= 2 rewards")
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if not scale:
        return centered
    var = sum(c * c for c in centered) / len(centered)
    std = math.sqrt(var)
    return [c / (std + eps) for c in centered]


# --------------------------------------------------------------------------
# Sample construction (episode -> masked token sequence + old logprobs)
# --------------------------------------------------------------------------

@dataclass
class RolloutSample:
    task_id: str
    group_index: int
    reward_total: float
    success: bool
    advantage: float
    input_ids: list[int]
    supervised_positions: list[int]
    old_logprobs: list[float]
    termination: str
    tool_call_count: int
    invalid_call_count: int


@dataclass
class RolloutBatchStats:
    rollouts: int = 0
    kept: int = 0
    dropped_policy_error: int = 0
    dropped_too_long: int = 0
    dropped_zero_variance: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    reward_sumsq: float = 0.0
    invalid_call_total: int = 0
    tool_call_total: int = 0
    turns_total: int = 0

    @property
    def reward_mean(self) -> float:
        return self.reward_sum / self.rollouts if self.rollouts else 0.0

    @property
    def reward_std(self) -> float:
        if not self.rollouts:
            return 0.0
        var = max(self.reward_sumsq / self.rollouts - self.reward_mean ** 2, 0.0)
        return math.sqrt(var)

    def merge(self, other: "RolloutBatchStats") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["reward_mean"] = round(self.reward_mean, 6)
        out["reward_std"] = round(self.reward_std, 6)
        return out


def episode_training_messages(task: Task, episode: Any, *,
                              compact_catalog: bool) -> list[dict[str, str]] | None:
    """Model-visible messages of an episode, ready for tokenization.

    - Trailing non-assistant messages (step-limit / loop-guard episodes end on
      a tool observation) are removed so the sequence ends where the policy
      last acted — those tokens are still what the policy produced.
    - ``policy_error`` episodes return None (infrastructure failure, not
      policy behavior; they must not shape gradients).
    """
    if episode.termination == "policy_error":
        return None
    messages = messages_from_episode(task, episode)
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 3:
        return None
    if compact_catalog:
        messages[0] = {"role": "system",
                       "content": compact_system_prompt(messages[0]["content"], episode.tools)}
    return messages


def token_logprobs_at(model: Any, input_ids: Any, positions: list[int]) -> Any:
    """log pi(y_t | y_<t) for each supervised position t (gradient-tracked).

    Position p's logprob comes from logits at p-1 (next-token prediction).
    """
    import torch
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    logits = outputs.logits[0].float()                      # [T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_ids = input_ids[0]
    idx = torch.tensor([p - 1 for p in positions], device=logits.device)
    gathered = log_probs.index_select(0, idx)               # [S, V]
    targets = token_ids[torch.tensor(positions, device=logits.device)].unsqueeze(-1)
    return gathered.gather(-1, targets).squeeze(-1)         # [S]


class GrpoTrainer:
    """Custom GRPO trainer for executor-grounded, verifier-rewarded tool use."""

    def __init__(self, settings: GrpoSettings, *,
                 policy_factory: Callable[[], Any] | None = None,
                 eval_policy_factory: Callable[[], Any] | None = None) -> None:
        settings.validate()
        self.settings = settings
        self._policy_factory_override = policy_factory
        self._eval_policy_factory_override = eval_policy_factory
        self._model = None
        self._tokenizer = None
        self._model_info: dict[str, Any] = {}
        self._executor: Executor | None = None
        self._optimizer = None
        self._history_path = Path(settings.output_dir) / "history.jsonl"
        self._step_counter = 0
        self._task_order_seed = settings.seed

    # -- lazy resource setup -------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        s = self.settings
        set_seed(s.seed)
        from types import SimpleNamespace
        # resolve_device/resolve_precision read .device/.precision (SftSettings
        # vocabulary); GrpoSettings spells precision `dtype` — adapt explicitly.
        device = resolve_device(SimpleNamespace(device=s.device))
        precision = resolve_precision(SimpleNamespace(precision=s.dtype), device)
        quantization_config = None
        if s.use_4bit:
            if device != "cuda":
                raise GrpoError("use_4bit requires CUDA; GRPO defaults to LoRA fp16/bf16")
            from transformers import BitsAndBytesConfig
            compute = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[precision]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True)
        logger.info("GRPO loading %s (adapter=%s, device=%s, precision=%s)",
                    s.model_id, s.adapter_id, device, precision)
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(s.model_id, trust_remote_code=s.trust_remote_code)
        kw: dict[str, Any] = {"trust_remote_code": s.trust_remote_code}
        if quantization_config is not None:
            kw["quantization_config"] = quantization_config
        else:
            kw[_model_dtype_kwarg_name()] = {"bf16": torch.bfloat16, "fp16": torch.float16,
                                             "fp32": torch.float32}[precision]
        base = AutoModelForCausalLM.from_pretrained(s.model_id, **kw)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if s.adapter_id:
            model = PeftModel.from_pretrained(base, s.adapter_id, is_trainable=True)
        else:
            logger.warning("No adapter_id: GRPO would start from the raw base model. "
                           "The intended initialization is the SFT checkpoint.")
            lora_config = LoraConfig(r=s.lora.r, lora_alpha=s.lora.alpha,
                                     lora_dropout=s.lora.dropout,
                                     target_modules=list(s.lora.target_modules),
                                     bias=s.lora.bias, task_type=TaskType.CAUSAL_LM)
            model = get_peft_model(base, lora_config)
        if s.gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
        model.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._model, self._tokenizer = model, tokenizer
        self._model_info = {"device": device, "precision": precision, "use_4bit": s.use_4bit,
                            "trainable_params": trainable,
                            "load_seconds": round(time.perf_counter() - t0, 2)}
        self._optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=s.learning_rate, weight_decay=s.weight_decay)

    def _executor_instance(self) -> Executor:
        if self._executor is None:
            from toolforge.config import EnvSettings
            self._executor = Executor(default_registry(), settings=EnvSettings())
        return self._executor

    def _rollout_policy(self, seed: int) -> Any:
        if self._policy_factory_override is not None:
            return self._policy_factory_override()
        s = self.settings
        return HuggingFacePolicy(
            s.model_id, tokenizer=self._tokenizer, model=self._model,
            max_new_tokens=s.max_new_tokens, do_sample=True, temperature=s.temperature,
            top_p=s.top_p, seed=seed, enable_thinking=s.enable_thinking,
            compact_catalog=s.compact_catalog)

    def _eval_policy(self) -> Any:
        if self._eval_policy_factory_override is not None:
            return self._eval_policy_factory_override()
        s = self.settings
        return HuggingFacePolicy(
            s.model_id, tokenizer=self._tokenizer, model=self._model,
            max_new_tokens=s.max_new_tokens, do_sample=False,
            enable_thinking=s.enable_thinking, compact_catalog=s.compact_catalog)

    # -- rollouts --------------------------------------------------------------

    def collect_group(self, task: Task, iteration: int) -> tuple[list[Any], list[Any], list[float]]:
        """G sampled episodes for one task + verifier verdicts + rewards."""
        s = self.settings
        executor = self._executor_instance()
        episodes, verdicts, rewards = [], [], []
        was_training = self._model.training if self._model is not None else False
        if self._model is not None:
            self._model.eval()  # generation without dropout; restored below
        try:
            for g in range(s.num_generations):
                digest = int(hashlib.sha256(
                    f"{task.task_id}|{g}|{iteration}".encode()).hexdigest()[:8], 16)
                rollout_seed = s.seed + 1_000_003 * iteration + digest % 9_973
                policy = self._rollout_policy(rollout_seed)
                set_sampling = getattr(policy, "set_sampling", None)
                if callable(set_sampling):
                    set_sampling(seed=rollout_seed)
                episode = executor.run(task, policy)
                verdict = verify_episode(task, episode)
                reward = compute_reward(episode, verdict, s.reward, max_steps=task.max_steps)
                episodes.append(episode)
                verdicts.append(verdict)
                rewards.append(reward.total)
        finally:
            if self._model is not None and was_training:
                self._model.train()
        return episodes, verdicts, rewards

    def build_samples(self, task: Task, episodes: list, verdicts: list,
                      rewards: list[float]) -> tuple[list[RolloutSample], RolloutBatchStats]:
        """Tokenize episodes into training samples with advantages + old logprobs."""
        import torch
        s = self.settings
        stats = RolloutBatchStats(rollouts=len(episodes))
        stats.successes = sum(1 for v in verdicts if v.success)
        stats.reward_sum = sum(rewards)
        stats.reward_sumsq = sum(r * r for r in rewards)
        advantages = group_advantages(rewards, scale=s.scale_rewards)
        if s.skip_zero_variance and stats.reward_std <= 1e-9:
            stats.dropped_zero_variance = len(episodes)
            return [], stats

        samples: list[RolloutSample] = []
        for g, (episode, verdict, reward_total, adv) in enumerate(
                zip(episodes, verdicts, rewards, advantages)):
            stats.invalid_call_total += episode.invalid_call_count
            stats.tool_call_total += episode.tool_call_count
            stats.turns_total += len(episode.trace)
            messages = episode_training_messages(
                task, episode, compact_catalog=s.compact_catalog)
            if messages is None:
                stats.dropped_policy_error += 1
                continue
            tok = tokenize_sft_messages(
                messages, self._tokenizer, max_seq_len=s.max_seq_len,
                chat_template_kwargs={"enable_thinking": s.enable_thinking})
            if tok is None:
                stats.dropped_too_long += 1
                continue
            positions = [i for i, l in enumerate(tok["labels"]) if l != IGNORE_INDEX and i > 0]
            if not positions:
                stats.dropped_too_long += 1
                continue
            ids = torch.tensor([tok["input_ids"]])
            with torch.no_grad():
                old_lp = token_logprobs_at(self._model, ids, positions).detach().cpu().tolist()
            samples.append(RolloutSample(
                task_id=task.task_id, group_index=g, reward_total=reward_total,
                success=verdict.success, advantage=adv, input_ids=tok["input_ids"],
                supervised_positions=positions, old_logprobs=old_lp,
                termination=episode.termination, tool_call_count=episode.tool_call_count,
                invalid_call_count=episode.invalid_call_count))
            stats.kept += 1
        return samples, stats

    # -- optimization ------------------------------------------------------------

    def optimization_pass(self, samples: list[RolloutSample]) -> dict[str, float]:
        """ppo_epochs passes of clipped-surrogate updates over the samples."""
        import torch
        s = self.settings
        assert self._model is not None and self._optimizer is not None
        rng = random.Random(s.seed + self._step_counter)
        totals = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_frac": 0.0,
                  "ratio_mean": 0.0, "steps": 0}
        for epoch in range(s.ppo_epochs):
            order = list(range(len(samples)))
            rng.shuffle(order)
            self._optimizer.zero_grad(set_to_none=True)
            for n, idx in enumerate(order, start=1):
                sample = samples[idx]
                ids = torch.tensor([sample.input_ids], device=self._device())
                positions = sample.supervised_positions
                new_lp = token_logprobs_at(self._model, ids, positions)
                old_lp = torch.tensor(sample.old_logprobs, device=new_lp.device, dtype=new_lp.dtype)
                adv = torch.tensor(sample.advantage, device=new_lp.device, dtype=new_lp.dtype)
                ratio = torch.exp(new_lp - old_lp)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - s.clip_eps, 1 + s.clip_eps) * adv
                pg_loss = -torch.min(unclipped, clipped).mean()
                kl_term = torch.zeros((), device=new_lp.device, dtype=new_lp.dtype)
                if s.beta_kl > 0:
                    kl_term = self._kl_to_reference(ids, positions, new_lp)
                loss = pg_loss + s.beta_kl * kl_term
                (loss / len(order)).backward()
                with torch.no_grad():
                    totals["loss"] += float(loss.detach())
                    totals["pg_loss"] += float(pg_loss.detach())
                    totals["kl"] += float(kl_term.detach())
                    frac = float(((ratio - 1).abs() > s.clip_eps).float().mean())
                    totals["clip_frac"] += frac
                    totals["ratio_mean"] += float(ratio.mean())
                    totals["steps"] += 1
                if n == len(order):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._model.parameters() if p.requires_grad],
                        s.max_grad_norm)
                    self._apply_lr_schedule()
                    self._optimizer.step()
                    self._step_counter += 1
        return {k: (v / max(totals["steps"], 1) if k != "steps" else v)
                for k, v in totals.items()}

    def _apply_lr_schedule(self) -> None:
        """Linear warmup over ``warmup_steps`` optimizer steps, then constant."""
        s = self.settings
        if s.warmup_steps <= 0:
            return
        scale = min(1.0, (self._step_counter + 1) / s.warmup_steps)
        for group in self._optimizer.param_groups:
            group["lr"] = s.learning_rate * scale

    def _kl_to_reference(self, ids: Any, positions: list[int], new_lp: Any) -> Any:
        """k3 KL estimator against the base model with adapters disabled."""
        import torch
        disable = getattr(self._model, "disable_adapter", None)
        if disable is None:
            raise GrpoError("beta_kl > 0 requires a PEFT model (adapter-disable reference)")
        with torch.no_grad():
            with disable():
                ref_lp = token_logprobs_at(self._model, ids, positions)
        delta = ref_lp.detach() - new_lp
        return (torch.exp(delta) - delta - 1).mean()

    def _device(self) -> Any:
        import torch
        p = next(self._model.parameters())
        return p.device if p.device.type != "meta" else torch.device("cpu")

    # -- evaluation / divergence ---------------------------------------------------

    def evaluate(self, eval_tasks: list[Task]) -> dict[str, Any]:
        executor = self._executor_instance()
        policy = self._eval_policy()
        successes = invalid = tool_calls = turns = 0
        was_training = self._model.training if self._model is not None else False
        if self._model is not None:
            self._model.eval()
        for task in eval_tasks:
            episode = executor.run(task, policy)
            verdict = verify_episode(task, episode)
            successes += int(verdict.success)
            invalid += episode.invalid_call_count
            tool_calls += episode.tool_call_count
            turns += len(episode.trace)
        if self._model is not None and was_training:
            self._model.train()
        n = max(len(eval_tasks), 1)
        return {"n_tasks": len(eval_tasks), "success_rate": successes / n,
                "invalid_per_task": invalid / n, "tool_calls_per_task": tool_calls / n,
                "turns_per_task": turns / n}

    def check_divergence(self, history: list[dict[str, Any]]) -> bool:
        """True when training reward is rising while validated accuracy is falling.

        Compares the two most recent evals: mean train reward since the previous
        eval increased, but eval success_rate dropped by more than
        ``divergence_threshold`` -> possible reward hacking; flagged loudly.
        """
        evals = [(h["iteration"], h["eval"]["success_rate"], h.get("window_reward_mean"))
                 for h in history if h.get("eval") and h.get("window_reward_mean") is not None]
        if len(evals) < 2:
            return False
        (_, acc0, rw0), (_, acc1, rw1) = evals[-2], evals[-1]
        return rw1 > rw0 and (acc0 - acc1) > self.settings.divergence_threshold

    # -- checkpointing ----------------------------------------------------------

    def _checkpoint_dir(self, iteration: int) -> Path:
        return Path(self.settings.output_dir) / f"checkpoint-iter{iteration:05d}"

    def save_checkpoint(self, iteration: int, extra: dict[str, Any]) -> Path:
        import torch
        ckpt = self._checkpoint_dir(iteration)
        ckpt.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(ckpt))
        torch.save({"optimizer": self._optimizer.state_dict(),
                    "iteration": iteration,
                    "step_counter": self._step_counter,
                    "rng_python": random.getstate(),
                    "rng_torch": torch.get_rng_state(),
                    "task_order_seed": self._task_order_seed},
                   ckpt / "training_state.pt")
        save_json_atomic(ckpt / "grpo_metadata.json", {**self._base_metadata(), **extra})
        self._prune_checkpoints()
        return ckpt

    def _prune_checkpoints(self) -> None:
        out = Path(self.settings.output_dir)
        ckpts = sorted(out.glob("checkpoint-iter*"))
        while len(ckpts) > max(self.settings.save_total_limit, 1):
            oldest = ckpts.pop(0)
            for f in oldest.iterdir():
                f.unlink()
            oldest.rmdir()
            logger.info("Pruned old checkpoint %s", oldest)

    def find_resume_checkpoint(self) -> Path | None:
        out = Path(self.settings.output_dir)
        candidates = sorted(out.glob("checkpoint-iter*/training_state.pt"))
        return candidates[-1].parent if candidates else None

    def _base_metadata(self) -> dict[str, Any]:
        s = self.settings
        return {
            "phase": "grpo", "grammar_version": GRAMMAR_VERSION,
            "settings": dataclasses.asdict(s), "model_info": self._model_info,
            "versions": library_versions(),
            "tasks_fingerprint": data_fingerprint(s.tasks_file)
            if Path(s.tasks_file).exists() else None,
        }

    # -- main loop -----------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        import torch
        s = self.settings
        self._ensure_model()
        rng = random.Random(s.seed)
        self._task_order_seed = s.seed

        all_tasks = load_tasks(s.tasks_file)
        train_tasks = tasks_by_split(all_tasks, s.split)
        if s.task_limit:
            train_tasks = sorted(train_tasks, key=lambda t: t.task_id)[:s.task_limit]
        if not train_tasks:
            raise GrpoError(f"No tasks in split {s.split!r} of {s.tasks_file}")
        eval_tasks = sorted(tasks_by_split(all_tasks, s.eval_split), key=lambda t: t.task_id)
        if s.eval_task_limit:
            eval_tasks = eval_tasks[:s.eval_task_limit]
        preflight_tasks(train_tasks + eval_tasks, self._executor_instance())

        start_iteration = 0
        history: list[dict[str, Any]] = []
        if s.resume:
            ckpt = self.find_resume_checkpoint()
            if ckpt is None:
                logger.info("resume requested, no checkpoint found; starting fresh")
            else:
                state = torch.load(ckpt / "training_state.pt", weights_only=False)
                self._load_adapter_from(ckpt)
                self._optimizer.load_state_dict(state["optimizer"])
                start_iteration = state["iteration"] + 1
                self._step_counter = state.get("step_counter", 0)
                random.setstate(state["rng_python"])
                torch.set_rng_state(state["rng_torch"])
                if self._history_path.exists():
                    history = [json.loads(l) for l in
                               self._history_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                logger.info("Resumed from %s (iteration %d)", ckpt, start_iteration)

        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        order = [t.task_id for t in train_tasks]
        rng.shuffle(order)
        by_id = {t.task_id: t for t in train_tasks}
        cursor = 0
        window_rewards: list[float] = []
        best_eval = max((h["eval"]["success_rate"] for h in history if h.get("eval")),
                        default=None)

        for iteration in range(start_iteration, s.total_iterations):
            t0 = time.perf_counter()
            if cursor + s.rollout_batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            batch = [by_id[order[cursor + i]] for i in range(s.rollout_batch_size)]
            cursor += s.rollout_batch_size

            all_samples: list[RolloutSample] = []
            iter_stats = RolloutBatchStats()
            for task in batch:
                episodes, verdicts, rewards = self.collect_group(task, iteration)
                window_rewards.extend(rewards)
                samples, stats = self.build_samples(task, episodes, verdicts, rewards)
                all_samples.extend(samples)
                iter_stats.merge(stats)

            train_metrics = {"loss": None}
            if all_samples:
                train_metrics = self.optimization_pass(all_samples)
            else:
                logger.info("Iteration %d: no trainable samples (all groups zero-variance "
                            "or dropped)", iteration)

            record: dict[str, Any] = {
                "iteration": iteration,
                "tasks": [t.task_id for t in batch],
                "rollouts": iter_stats.to_dict(),
                "train": train_metrics,
                "window_reward_mean": (sum(window_rewards) / len(window_rewards)
                                       if window_rewards else None),
                "seconds": round(time.perf_counter() - t0, 2),
            }

            if s.eval_every and (iteration + 1) % s.eval_every == 0 and eval_tasks:
                record["eval"] = self.evaluate(eval_tasks)
                if best_eval is None or record["eval"]["success_rate"] > best_eval:
                    best_eval = record["eval"]["success_rate"]
                record["best_eval_success_rate"] = best_eval

            history.append(record)
            if s.log_every and (iteration + 1) % s.log_every == 0:
                ev = record.get("eval")
                if ev and self.check_divergence(history):
                    logger.warning(
                        "DIVERGENCE WARNING: training reward is rising while validated task "
                        "accuracy is falling (possible reward hacking). Inspect history.jsonl "
                        "and the latest predictions before trusting this checkpoint.")
                    record["divergence_warning"] = True
                self._append_history(record)
                logger.info(
                    "iter %d | reward mean %.3f std %.3f | success %d/%d | invalid %d | "
                    "kept %d/%d | loss %s%s%s",
                    iteration, iter_stats.reward_mean, iter_stats.reward_std,
                    iter_stats.successes, iter_stats.rollouts, iter_stats.invalid_call_total,
                    iter_stats.kept, iter_stats.rollouts,
                    f"{train_metrics.get('loss', 0):.4f}" if train_metrics.get("loss") is not None else "n/a",
                    f" | EVAL success {ev['success_rate']:.3f}" if ev else "",
                    " | DIVERGENCE WARNING" if record.get("divergence_warning") else "")
            if record.get("eval"):
                window_rewards = []  # window = rewards since the last validated eval

            if s.save_every and (iteration + 1) % s.save_every == 0:
                self.save_checkpoint(iteration, {"iteration": iteration,
                                                 "best_eval_success_rate": best_eval})

        final_dir = self.save_checkpoint(s.total_iterations - 1,
                                         {"final": True, "best_eval_success_rate": best_eval})
        summary = {
            "final_checkpoint": str(final_dir),
            "iterations_run": len(history),
            "best_eval_success_rate": best_eval,
            "total_rollouts": sum(h["rollouts"]["rollouts"] for h in history),
            "total_successes": sum(h["rollouts"]["successes"] for h in history),
            "divergence_warnings": sum(1 for h in history if h.get("divergence_warning")),
        }
        save_json_atomic(Path(s.output_dir) / "run_metadata.json",
                         {**self._base_metadata(), "summary": summary})
        logger.info("GRPO complete: %s", summary)
        return summary

    def _load_adapter_from(self, ckpt: Path) -> None:
        """Point the trainable model at a resumed checkpoint's adapter weights."""
        import torch
        from peft import PeftModel, set_peft_model_state_dict
        if not isinstance(self._model, PeftModel):
            raise GrpoError("Resume requires a PEFT-wrapped model")
        adapter_files = sorted(ckpt.glob("adapter_model.*"))
        if not adapter_files:
            raise GrpoError(f"No adapter weights in checkpoint {ckpt}")
        if adapter_files[0].suffix in (".bin", ".pt"):
            state = torch.load(adapter_files[0], map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            state = load_file(str(adapter_files[0]))
        # peft's loader inserts the active adapter name into each key; a raw
        # load_state_dict would silently mismatch ("lora_A.weight" vs
        # "lora_A.default.weight").
        result = set_peft_model_state_dict(self._model, state)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise GrpoError(f"Unexpected keys when resuming adapter: {unexpected[:5]}")
        loaded = len(state)
        logger.info("Resumed adapter weights from %s (%d tensors)", adapter_files[0].name, loaded)

    def _append_history(self, record: dict[str, Any]) -> None:
        with self._history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
