"""Model-agnostic policy interfaces for evaluation and benchmarking.

A *policy* maps a :class:`~toolforge.executor.PolicyContext` to raw text — the
same contract the executor drives in tests, so evaluation, trajectory
collection (Phase 3), GRPO rollouts (Phase 4), and the demo (Phase 6) all
share one serialization path.

Shipped policies:

- :class:`HuggingFacePolicy` — base model + optional PEFT adapter via
  transformers. torch/transformers are imported lazily: the core package stays
  CPU-only and importable without the training stack. For unit tests, inject
  ``tokenizer=`` and ``generate_fn=`` and no heavy dependency is touched.
- :class:`RandomPolicy` — schema-aware random baseline. Exercises the full
  validation/execution path on CPU and provides an honest lower-bound
  reference; also used to smoke-test the evaluation harness.
- :class:`OpenAICompatiblePolicy` — optional external teacher/API evaluation
  (works with any OpenAI-compatible chat endpoint). Disabled unless an API key
  env var is set; never required by the core project.

``tools_fingerprint`` pins the exact tool catalog + grammar version a policy
faced, so runs with incompatible serializations are detected instead of
silently compared.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from toolforge import get_logger
from toolforge.executor import PolicyContext, ScriptedPolicy  # re-export for callers
from toolforge.schemas import GRAMMAR_VERSION, ToolForgeError, ToolSpec

logger = get_logger("inference")

__all__ = [
    "GenerationStats", "PolicyError", "StatsTrackingPolicy", "HuggingFacePolicy",
    "RandomPolicy", "OpenAICompatiblePolicy", "ScriptedPolicy",
    "tools_fingerprint", "policy_id",
]


class PolicyError(ToolForgeError):
    """Raised when a policy cannot serve requests (missing deps/keys, API errors)."""


@dataclass
class GenerationStats:
    """Token/latency accounting for one generation call (None = unavailable)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_seconds: float | None = None      # time to first token, streaming only
    generate_seconds: float = 0.0


class StatsTrackingPolicy:
    """Mixin aggregating :class:`GenerationStats` across calls.

    The evaluation runner calls :meth:`reset_stats` before each task (when
    available) so token/latency totals can be attributed per task.
    """

    def _init_stats(self) -> None:
        self.reset_stats()
        self.load_seconds: float | None = None

    def reset_stats(self) -> None:
        self.stats = GenerationStats()
        self.call_count = 0
        self.last_call_stats: GenerationStats | None = None

    def _record_stats(self, gs: GenerationStats) -> None:
        self.last_call_stats = gs
        self.call_count += 1
        self.stats.prompt_tokens = _add_opt(self.stats.prompt_tokens, gs.prompt_tokens)
        self.stats.completion_tokens = _add_opt(self.stats.completion_tokens, gs.completion_tokens)
        self.stats.generate_seconds += gs.generate_seconds
        if gs.ttft_seconds is not None:
            self.stats.ttft_seconds = gs.ttft_seconds  # most recent measurable TTFT


def _add_opt(a: int | None, b: int | None) -> int | None:
    if b is None:
        return a
    return (a or 0) + b


def tools_fingerprint(specs: list[ToolSpec]) -> str:
    """Stable hash of the model-visible tool catalog + grammar version."""
    payload = GRAMMAR_VERSION + "\n" + json.dumps(
        sorted((s.to_dict() for s in specs), key=lambda d: d["name"]),
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def policy_id(policy: Any) -> str:
    """Human-readable identifier recorded in every evaluation run."""
    pid = getattr(policy, "policy_id", None)
    if pid:
        return str(pid)
    return type(policy).__name__


# --------------------------------------------------------------------------
# Hugging Face policy
# --------------------------------------------------------------------------

GenerateFn = Callable[[str, dict[str, Any]], tuple[str, int | None, int | None]]
"""Injected generation function ``(prompt_text, gen_kwargs) -> (text, prompt_tokens, completion_tokens)``."""


class HuggingFacePolicy(StatsTrackingPolicy):
    """Policy backed by a transformers causal LM (+ optional PEFT adapter).

    Serialization: ``tokenizer.apply_chat_template(messages, add_generation_prompt=True)``
    over the executor's system/user/assistant/tool messages — the same grammar
    used by training (Phase 3) and Phase 1 tests. Output text is the decoded
    *new* tokens only; the executor's parser extracts the single JSON action.
    """

    def __init__(self, model_id: str, adapter_id: str | None = None, *,
                 device: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 512, do_sample: bool = False,
                 temperature: float = 0.7, top_p: float = 0.95,
                 seed: int | None = None, enable_thinking: bool = False,
                 trust_remote_code: bool = False, compact_catalog: bool = False,
                 tokenizer: Any = None, model: Any = None,
                 generate_fn: GenerateFn | None = None) -> None:
        self._init_stats()
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.device_request = device
        self.dtype_request = dtype
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.trust_remote_code = trust_remote_code
        self.compact_catalog = compact_catalog
        self._tokenizer = tokenizer
        self._model = model
        self._generate_fn = generate_fn
        self._loaded = False
        self.device: str | None = None
        self.dtype: str | None = None
        self.policy_id = (f"hf:{model_id}" + (f"+adapter:{os.path.basename(adapter_id.rstrip('/'))}"
                                              if adapter_id else ""))

    # -- loading ---------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if self._tokenizer is None or (self._model is None and self._generate_fn is None):
            self._load_transformers()
        self._loaded = True

    def _load_transformers(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PolicyError(
                "HuggingFacePolicy requires torch and transformers (see requirements-train.txt, "
                f"Phase 3): {exc}") from exc
        t0 = time.perf_counter()
        device = self.device_request
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self.dtype_request
        if dtype == "auto":
            if device == "cuda":
                major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
                dtype = "bfloat16" if major >= 8 else "float16"  # T4 (7.x) has no fast bf16
            else:
                dtype = "float32"
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        logger.info("Loading %s (device=%s, dtype=%s, adapter=%s)",
                    self.model_id, device, dtype, self.adapter_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, trust_remote_code=self.trust_remote_code)
        if self.adapter_id:
            from peft import PeftModel  # lazy: only needed with adapters
            model = PeftModel.from_pretrained(model, self.adapter_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = model.to(device).eval()
        self.device, self.dtype = device, dtype
        self.load_seconds = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", self.load_seconds)

    # -- generation --------------------------------------------------------

    def set_sampling(self, *, do_sample: bool | None = None, seed: int | None = None,
                     temperature: float | None = None, top_p: float | None = None) -> None:
        """Adjust sampling for repeat-runs without rebuilding the policy."""
        if do_sample is not None:
            self.do_sample = do_sample
        if seed is not None:
            self.seed = seed
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p

    def _gen_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs

    def render_prompt(self, context: PolicyContext) -> str:
        self._load()
        messages = context.messages
        if self.compact_catalog and messages and messages[0]["role"] == "system":
            # Must match the training-side rendering for models SFT'd on
            # compacted catalogs (see data_generation.compact_system_prompt).
            from toolforge.data_generation import compact_system_prompt
            messages = list(messages)
            messages[0] = {"role": "system",
                           "content": compact_system_prompt(messages[0]["content"],
                                                            context.tools)}
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": self.enable_thinking})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            # Templates without thinking-mode support reject the kwarg; fall back
            # once, loudly, so the difference is visible in logs.
            logger.warning("Chat template rejected enable_thinking kwarg (%s); retrying without it", exc)
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, context: PolicyContext) -> str:
        self._load()
        prompt = self.render_prompt(context)
        gen_kwargs = self._gen_kwargs()
        t0 = time.perf_counter()
        if self._generate_fn is not None and self._model is None:
            text, p_tok, c_tok = self._generate_fn(prompt, gen_kwargs)
        else:
            text, p_tok, c_tok = self._generate_torch(prompt, gen_kwargs)
        gs = GenerationStats(prompt_tokens=p_tok, completion_tokens=c_tok,
                             generate_seconds=time.perf_counter() - t0)
        self._record_stats(gs)
        return text

    def _generate_torch(self, prompt: str, gen_kwargs: dict[str, Any]) -> tuple[str, int | None, int | None]:
        import torch
        assert self._model is not None and self._tokenizer is not None
        if self.do_sample and self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        eos = self._tokenizer.eos_token_id
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, **gen_kwargs,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos)
        new_ids = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, prompt_len, int(new_ids.shape[0])

    def generate_with_ttft(self, context: PolicyContext) -> tuple[str, GenerationStats]:
        """Streaming generation measuring time-to-first-token (benchmarking only).

        Falls back to non-streaming generation when the transformers streamer
        is unavailable; ``ttft_seconds`` is then ``None`` rather than faked.
        """
        self._load()
        prompt = self.render_prompt(context)
        gen_kwargs = self._gen_kwargs()
        if self._model is None:
            t0 = time.perf_counter()
            text, p_tok, c_tok = (self._generate_fn(prompt, gen_kwargs)
                                  if self._generate_fn else self._generate_torch(prompt, gen_kwargs))
            gs = GenerationStats(p_tok, c_tok, None, time.perf_counter() - t0)
            self._record_stats(gs)
            return text, gs
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise PolicyError(f"TTFT measurement requires transformers: {exc}") from exc
        assert self._tokenizer is not None
        if self.do_sample and self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=dict(**inputs, **gen_kwargs,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id, streamer=streamer),
            daemon=True)
        t0 = time.perf_counter()
        thread.start()
        chunks: list[str] = []
        ttft: float | None = None
        for chunk in streamer:
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(chunk)
        thread.join()
        total = time.perf_counter() - t0
        text = "".join(chunks)
        c_tok = len(self._tokenizer.encode(text, add_special_tokens=False)) if text else 0
        gs = GenerationStats(prompt_len, c_tok, ttft, total)
        self._record_stats(gs)
        return text, gs


# --------------------------------------------------------------------------
# Random baseline policy
# --------------------------------------------------------------------------

_STRING_POOL = ["2026-03-03", "2026-03-04", "10:00", "11:00", "example", "note",
                "alice@example.com", "task-item", "/notes/x.md", "high", "call", "item-1"]


class RandomPolicy(StatsTrackingPolicy):
    """Schema-aware random baseline (deterministic given ``seed``).

    Emits valid tool calls with schema-sampled arguments for
    ``max_tool_calls`` turns, then a final answer. ``malformed_rate`` injects
    unparseable output to exercise invalid-JSON metrics. This policy is a
    *lower-bound reference and harness exerciser*, not a serious agent.
    """

    def __init__(self, seed: int = 0, max_tool_calls: int = 3, malformed_rate: float = 0.0,
                 clarify_rate: float = 0.05) -> None:
        self._init_stats()
        self._rng = random.Random(seed)
        self.max_tool_calls = max_tool_calls
        self.malformed_rate = malformed_rate
        self.clarify_rate = clarify_rate
        self._calls_this_episode = 0
        self.policy_id = f"random:seed={seed}"

    def __call__(self, context: PolicyContext) -> str:
        if context.step == 0:
            self._calls_this_episode = 0
        t0 = time.perf_counter()
        if self._rng.random() < self.malformed_rate:
            out = "I think maybe { not valid json at all"
        elif self._calls_this_episode >= self.max_tool_calls or not context.tools:
            out = json.dumps({"action": "final_answer",
                              "content": "Random baseline finished; no meaningful result."})
        elif self._rng.random() < self.clarify_rate:
            out = json.dumps({"action": "clarify", "question": "Could you clarify the request?"})
        else:
            tool = self._rng.choice(context.tools)
            args = sample_arguments(tool.get("parameters", {}), self._rng)
            self._calls_this_episode += 1
            out = json.dumps({"action": "tool_call", "tool": tool["name"], "arguments": args})
        self._record_stats(GenerationStats(None, None, None, time.perf_counter() - t0))
        return out


def sample_arguments(schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Sample a schema-valid arguments object (used by RandomPolicy)."""
    args: dict[str, Any] = {}
    if schema.get("type") != "object":
        return args
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, prop in properties.items():
        if name not in required and rng.random() < 0.5:
            continue
        args[name] = _sample_value(prop, rng)
    return args


def _sample_value(prop: dict[str, Any], rng: random.Random) -> Any:
    if "enum" in prop:
        return rng.choice(prop["enum"])
    if "const" in prop:
        return prop["const"]
    ptype = prop.get("type", "string")
    if isinstance(ptype, list):
        ptype = ptype[0]
    pattern = prop.get("pattern", "")
    if ptype == "string":
        if r"\d{4}-\d{2}-\d{2}" in pattern and "T" not in pattern:
            return f"2026-03-{rng.randrange(2, 9):02d}"
        if r"[01]\d|2[0-3]" in pattern:
            return f"{rng.randrange(9, 17):02d}:00"
        if pattern.startswith("^/"):
            return rng.choice(["/notes/x.md", "/reports/y.md", "/specs/z.md"])
        return rng.choice(_STRING_POOL)
    if ptype == "integer":
        return rng.randint(int(prop.get("minimum", 0)), max(int(prop.get("minimum", 0)) + 1,
                                                            int(prop.get("maximum", 100))))
    if ptype == "number":
        return round(rng.uniform(float(prop.get("minimum", 0.0)),
                                 float(prop.get("maximum", 100.0))), 2)
    if ptype == "boolean":
        return rng.random() < 0.5
    if ptype == "array":
        return [_sample_value(prop.get("items", {"type": "string"}), rng)]
    if ptype == "object":
        return sample_arguments(prop, rng)
    return None


# --------------------------------------------------------------------------
# OpenAI-compatible API policy (optional teacher)
# --------------------------------------------------------------------------

class OpenAICompatiblePolicy(StatsTrackingPolicy):
    """Policy for any OpenAI-compatible /chat/completions endpoint.

    Used only when the user explicitly configures an API key env var; the core
    project never requires it. Implemented with stdlib urllib to avoid an SDK
    dependency and version drift.
    """

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 120.0,
                 max_new_tokens: int = 512, temperature: float = 0.0,
                 urlopen_fn: Callable[[Any], Any] | None = None) -> None:
        self._init_stats()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._urlopen = urlopen_fn
        self.policy_id = f"api:{model}@{self.base_url}"

    def __call__(self, context: PolicyContext) -> str:
        import urllib.error
        import urllib.request
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise PolicyError(
                f"{self.api_key_env} is not set; external API policies are optional and "
                "require an explicitly configured key (never hardcoded).")
        payload = {
            "model": self.model,
            "messages": context.messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST")
        opener = self._urlopen or urllib.request.urlopen
        t0 = time.perf_counter()
        try:
            with opener(request) as response:  # noqa: S310 (https endpoint, user-configured)
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise PolicyError(f"API request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PolicyError(f"API request failed: {exc.reason}") from exc
        elapsed = time.perf_counter() - t0
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PolicyError(f"Unexpected API response shape: {str(body)[:300]}") from exc
        usage = body.get("usage") or {}
        self._record_stats(GenerationStats(usage.get("prompt_tokens"),
                                           usage.get("completion_tokens"), None, elapsed))
        return text if isinstance(text, str) else json.dumps(text)
