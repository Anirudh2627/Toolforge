"""Tests for policy interfaces: fingerprints, HF policy (with injected fakes),
random baseline, and the OpenAI-compatible client. No model downloads.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from toolforge.executor import PolicyContext
from toolforge.inference import (
    GenerationStats,
    HuggingFacePolicy,
    OpenAICompatiblePolicy,
    PolicyError,
    RandomPolicy,
    policy_id,
    sample_arguments,
    tools_fingerprint,
)
from toolforge.schemas import GRAMMAR_VERSION, parse_action, validate_against_schema


def make_context(request: str = "Create an event.", tools: list[dict] | None = None,
                 step: int = 0) -> PolicyContext:
    return PolicyContext(
        request=request,
        tools=tools if tools is not None else [],
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": request}],
        step=step, remaining_steps=8)


class TestToolsFingerprint:
    def test_stable_and_order_independent(self, registry):
        specs = registry.tool_specs(["calendar"])
        assert tools_fingerprint(specs) == tools_fingerprint(list(reversed(specs)))

    def test_changes_with_catalog(self, registry):
        cal = registry.tool_specs(["calendar"])
        eml = registry.tool_specs(["email"])
        assert tools_fingerprint(cal) != tools_fingerprint(eml)

    def test_grammar_version_participates(self, registry, monkeypatch):
        import toolforge.inference as inf
        specs = registry.tool_specs(["calendar"])
        before = tools_fingerprint(specs)
        monkeypatch.setattr(inf, "GRAMMAR_VERSION", GRAMMAR_VERSION + "-X")
        assert tools_fingerprint(specs) != before


class FakeTokenizer:
    """Minimal stand-in exercising the chat-template code path."""

    def __init__(self, reject_template_kwargs: bool = False):
        self.reject_template_kwargs = reject_template_kwargs
        self.last_kwargs = "unset"
        self.pad_token_id = 1
        self.eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        assert tokenize is False and add_generation_prompt is True
        if chat_template_kwargs is not None:
            if self.reject_template_kwargs:
                raise TypeError("unexpected keyword 'chat_template_kwargs'")
            self.last_kwargs = chat_template_kwargs
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestHuggingFacePolicyInjected:
    def make_policy(self, tokenizer=None, outputs=None):
        calls = []

        def fake_generate(prompt, gen_kwargs):
            calls.append((prompt, dict(gen_kwargs)))
            text = (outputs or ["{}"]).pop(0)
            return text, len(prompt.split()), 7

        policy = HuggingFacePolicy(
            "fake/model", tokenizer=tokenizer or FakeTokenizer(), generate_fn=fake_generate,
            max_new_tokens=64, do_sample=False)
        return policy, calls

    def test_call_returns_text_and_records_stats(self):
        policy, calls = self.make_policy(outputs=['{"action": "final_answer", "content": "ok"}'])
        out = policy(make_context("Do the thing"))
        assert parse_action(out).kind.value == "final_answer"
        assert policy.stats.completion_tokens == 7
        assert policy.stats.prompt_tokens == len(calls[0][0].split())
        assert policy.call_count == 1

    def test_prompt_contains_roles_and_request(self):
        policy, calls = self.make_policy()
        policy(make_context("Book the room"))
        prompt = calls[0][0]
        assert "system: sys" in prompt and "user: Book the room" in prompt
        assert prompt.endswith("assistant:")

    def test_thinking_kwarg_passed_and_fallback(self):
        tok = FakeTokenizer()
        policy, _ = self.make_policy(tokenizer=tok)
        policy(make_context())
        assert tok.last_kwargs == {"enable_thinking": False}

        strict_tok = FakeTokenizer(reject_template_kwargs=True)
        policy2, _ = self.make_policy(tokenizer=strict_tok)
        out = policy2(make_context())  # must not raise; falls back loudly (logged warning)
        assert isinstance(out, str)

    def test_gen_kwargs_respect_sampling(self):
        policy, calls = self.make_policy()
        policy.set_sampling(do_sample=True, temperature=0.9, top_p=0.8, seed=5)
        policy(make_context())
        kwargs = calls[0][1]
        assert kwargs["do_sample"] is True and kwargs["temperature"] == 0.9
        assert kwargs["top_p"] == 0.8 and kwargs["max_new_tokens"] == 64
        assert policy.seed == 5

    def test_greedy_kwargs_exclude_sampling_params(self):
        policy, calls = self.make_policy()
        policy(make_context())
        assert "temperature" not in calls[0][1]

    def test_policy_id_and_load_without_transformers(self):
        policy, _ = self.make_policy()
        assert policy.policy_id == "hf:fake/model"
        assert policy_id(policy) == "hf:fake/model"
        policy._load()  # injected tokenizer+generate_fn: must not import torch

    def test_missing_dependency_raises_policy_error(self, monkeypatch):
        policy = HuggingFacePolicy("real/model")  # nothing injected
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name in ("torch", "transformers") or name.startswith(("torch.", "transformers.")):
                raise ImportError("no torch in this sandbox")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(PolicyError, match="requirements-train"):
            policy(make_context())

    def test_generate_with_ttft_nonstream_path(self):
        policy, _ = self.make_policy(outputs=["hello"])
        text, stats = policy.generate_with_ttft(make_context())
        assert text == "hello" and stats.ttft_seconds is None
        assert isinstance(stats, GenerationStats)

    def test_reset_stats(self):
        policy, _ = self.make_policy(outputs=["a", "b"])
        policy(make_context())
        policy.reset_stats()
        assert policy.call_count == 0 and policy.stats.completion_tokens is None


class TestRandomPolicy:
    def test_deterministic_given_seed(self, registry):
        tools = [s.to_dict() for s in registry.tool_specs(["calendar"])]
        contexts = [make_context(tools=tools, step=i) for i in range(6)]
        policy1 = RandomPolicy(seed=7)
        outs1 = [policy1(c) for c in contexts]
        policy2 = RandomPolicy(seed=7)
        outs2 = [policy2(c) for c in contexts]
        assert outs1 == outs2

    def test_outputs_are_valid_json_actions(self, registry):
        tools = [s.to_dict() for s in registry.tool_specs()]
        policy = RandomPolicy(seed=3, max_tool_calls=3)
        for step in range(5):
            out = policy(make_context(tools=tools, step=step))
            action = parse_action(out)
            assert not hasattr(action, "category"), out  # ParseFailure has .category

    def test_malformed_rate(self):
        policy = RandomPolicy(seed=1, malformed_rate=1.0)
        out = policy(make_context(tools=[], step=0))
        assert parse_action(out).category == "invalid_json"

    def test_terminates_with_final_answer(self, registry):
        tools = [s.to_dict() for s in registry.tool_specs(["todo"])]
        policy = RandomPolicy(seed=5, max_tool_calls=2)
        last = None
        for step in range(6):
            last = policy(make_context(tools=tools, step=step))
        assert parse_action(last).kind.value == "final_answer"

    def test_sampled_arguments_are_schema_valid(self, registry):
        import random as _random
        rng = _random.Random(0)
        for spec in registry.tool_specs():
            for _ in range(20):
                args = sample_arguments(spec.parameters, rng)
                issues = validate_against_schema(args, spec.parameters)
                assert not issues, (spec.name, args, issues)

    def test_stats_tracked(self):
        policy = RandomPolicy(seed=2)
        policy(make_context(tools=[], step=0))
        assert policy.call_count == 1
        assert policy.stats.generate_seconds >= 0.0


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestOpenAICompatiblePolicy:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("TOOLFORGE_TEST_KEY", raising=False)
        policy = OpenAICompatiblePolicy("gpt-test", api_key_env="TOOLFORGE_TEST_KEY")
        with pytest.raises(PolicyError, match="never hardcoded"):
            policy(make_context())

    def test_request_payload_and_response_parsing(self, monkeypatch):
        monkeypatch.setenv("TOOLFORGE_TEST_KEY", "sk-test-123")
        captured = {}

        def fake_urlopen(request):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse({
                "choices": [{"message": {"content": '{"action": "final_answer", "content": "hi"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5}})

        policy = OpenAICompatiblePolicy(
            "gpt-test", base_url="https://example.test/v1/", api_key_env="TOOLFORGE_TEST_KEY",
            max_new_tokens=99, temperature=0.1, urlopen_fn=fake_urlopen)
        out = policy(make_context("Do it", tools=[]))
        assert parse_action(out).kind.value == "final_answer"
        assert captured["url"] == "https://example.test/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-test-123"
        assert captured["payload"]["model"] == "gpt-test"
        assert captured["payload"]["max_tokens"] == 99
        assert captured["payload"]["temperature"] == 0.1
        assert policy.stats.prompt_tokens == 11 and policy.stats.completion_tokens == 5
        assert policy.policy_id.startswith("api:gpt-test@")

    def test_http_error_raises_policy_error(self, monkeypatch):
        monkeypatch.setenv("TOOLFORGE_TEST_KEY", "sk-test")

        def fake_urlopen(request):
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate limited", {},
                io.BytesIO(b'{"error": "slow down"}'))

        policy = OpenAICompatiblePolicy("m", api_key_env="TOOLFORGE_TEST_KEY",
                                        urlopen_fn=fake_urlopen)
        with pytest.raises(PolicyError, match="HTTP 429"):
            policy(make_context())

    def test_bad_response_shape_raises(self, monkeypatch):
        monkeypatch.setenv("TOOLFORGE_TEST_KEY", "sk-test")
        policy = OpenAICompatiblePolicy(
            "m", api_key_env="TOOLFORGE_TEST_KEY",
            urlopen_fn=lambda req: _FakeResponse({"unexpected": True}))
        with pytest.raises(PolicyError, match="Unexpected API response"):
            policy(make_context())


def test_policy_id_never_contains_secret(monkeypatch):
    monkeypatch.setenv("TOOLFORGE_TEST_KEY", "SUPERSECRET-VALUE")
    policy = OpenAICompatiblePolicy("m", api_key_env="TOOLFORGE_TEST_KEY")
    assert "SUPERSECRET" not in policy.policy_id
    assert policy.policy_id == "api:m@https://api.openai.com/v1"
