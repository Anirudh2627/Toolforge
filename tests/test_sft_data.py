"""Tests for SFT serialization: assistant-only loss masks, truncation policy,
record filtering. Uses a deterministic whitespace tokenizer with a ChatML-like
template (offset-mapping capable), so masks can be asserted token-exactly
without model downloads. A Qwen3-style artifact template proves the masking
tolerates position-dependent rendering.
"""

from __future__ import annotations

import pytest

from toolforge.sft_data import (
    IGNORE_INDEX,
    SftSerializationError,
    build_sft_records,
    load_sft_records,
    save_sft_records,
    tokenize_sft_file,
    tokenize_sft_messages,
    trajectory_to_sft_record,
)

PAD, IM_START, IM_END, UNK = "<|pad|>", "<|im_start|>", "<|im_end|>", "<|unk|>"
SPECIALS = [PAD, IM_START, IM_END, UNK]
ROLES = ["system", "user", "assistant", "tool"]


class TinyTokenizer:
    """Whitespace tokenizer with a ChatML-like, anchor-stable template and
    offset-mapping support (like every fast HF tokenizer)."""

    def __init__(self, extra_words=(), support_offsets: bool = True):
        self.vocab = {w: i for i, w in enumerate(SPECIALS + ROLES)}
        for word in extra_words:
            self._add(word)
        self.support_offsets = support_offsets
        self.additional_special_tokens = [IM_START, IM_END]
        self.pad_token_id = 0
        self.eos_token_id = self.vocab[IM_END]
        self.pad_token = PAD
        self.eos_token = IM_END

    def _add(self, word: str) -> None:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)

    def _ids(self, text: str) -> list[int]:
        unk = self.vocab[UNK]
        return [self.vocab.get(w, unk) for w in text.split()]

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 return_offsets_mapping=False):
        ids, offsets = [], []
        pos = 0
        for word in text.split():
            start = text.find(word, pos)
            end = start + len(word)
            ids.append(self.vocab.get(word, self.vocab[UNK]))
            offsets.append((start, end))
            pos = end
        out = {"input_ids": ids}
        if return_offsets_mapping and self.support_offsets:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        parts = [f"{IM_START} {m['role']} {m['content']} {IM_END}" for m in messages]
        text = " ".join(parts)
        if add_generation_prompt:
            text += f" {IM_START} assistant"
        return text

    # convenience for tests
    def supervised_text(self, out: dict) -> str:
        return " ".join(self._word(i) for i, l in zip(out["input_ids"], out["labels"])
                        if l != IGNORE_INDEX)

    def _word(self, token_id: int) -> str:
        for w, i in self.vocab.items():
            if i == token_id:
                return w
        return UNK


class NoOffsetTokenizer(TinyTokenizer):
    """Lacks offset_mapping -> exercises the strict prefix-diff fallback."""

    def __init__(self, extra_words=()):
        super().__init__(extra_words, support_offsets=False)


class BrokenTemplateTokenizer(TinyTokenizer):
    """Violates the anchor property (different header in generation prompts);
    with offsets unavailable the fallback must refuse to mask."""

    def __init__(self, extra_words=()):
        super().__init__(extra_words, support_offsets=False)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        text = super().apply_chat_template(messages, tokenize, False, chat_template_kwargs)
        if add_generation_prompt:
            text += " BROKENHEADER assistant"
        return text


class ArtifactTemplateTokenizer(TinyTokenizer):
    """Mimics Qwen3: injects an ARTIFACT marker before the LAST assistant
    message's content when rendering the full conversation (position-dependent
    rendering). Supervised spans must exclude the artifact."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        parts = []
        last = len(messages) - 1
        for i, m in enumerate(messages):
            content = m["content"]
            if m["role"] == "assistant" and i == last and not add_generation_prompt:
                content = "ARTIFACT " + content
            parts.append(f"{IM_START} {m['role']} {content} {IM_END}")
        text = " ".join(parts)
        if add_generation_prompt:
            text += f" {IM_START} assistant"
        return text


SAMPLE_MESSAGES = [
    {"role": "system", "content": "you are toolforge"},
    {"role": "user", "content": "create event sync tomorrow"},
    {"role": "assistant", "content": "calling calendar create now"},
    {"role": "tool", "content": "ok true event created"},
    {"role": "assistant", "content": "done created the event"},
]

ALL_WORDS = ("you are toolforge create event sync tomorrow calling calendar now "
             "ok true created done the").split()


def expected_supervised_words(tokenizer: TinyTokenizer) -> list[int]:
    """content words + end-of-turn per assistant message."""
    ids = []
    for m in SAMPLE_MESSAGES:
        if m["role"] == "assistant":
            ids.extend(tokenizer._ids(m["content"]))
            ids.append(tokenizer.vocab[IM_END])
    return ids


class TestMasking:
    def test_labels_cover_exactly_assistant_spans(self):
        tok = TinyTokenizer(extra_words=ALL_WORDS)
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        supervised = [i for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX]
        assert supervised == expected_supervised_words(tok)
        assert all(i == l for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX)

    def test_non_assistant_tokens_are_masked(self):
        tok = TinyTokenizer(extra_words=ALL_WORDS)
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        for word in ("toolforge", "tomorrow", "true", IM_START):
            wid = tok.vocab[word]
            positions = [k for k, i in enumerate(out["input_ids"]) if i == wid]
            assert positions, word
            assert all(out["labels"][k] == IGNORE_INDEX for k in positions), word

    def test_input_is_full_render(self):
        tok = TinyTokenizer()
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        rendered = tok.apply_chat_template(SAMPLE_MESSAGES, tokenize=False,
                                           add_generation_prompt=False)
        assert len(out["input_ids"]) == len(rendered.split())
        assert out["attention_mask"] == [1] * len(out["input_ids"])
        assert sum(1 for l in out["labels"] if l != IGNORE_INDEX) < len(out["labels"])

    def test_end_of_turn_token_supervised(self):
        tok = TinyTokenizer(extra_words=ALL_WORDS)
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        eot = tok.vocab[IM_END]
        spans, current, prev = [], [], None
        for i, l in enumerate(out["labels"]):
            if l != IGNORE_INDEX:
                if prev is not None and i != prev + 1:
                    spans.append(current)
                    current = []
                current.append(l)
                prev = i
        if current:
            spans.append(current)
        assert len(spans) == 2
        assert all(span[-1] == eot for span in spans)

    def test_qwen_style_artifact_excluded_from_supervision(self):
        tok = ArtifactTemplateTokenizer(extra_words=ALL_WORDS)
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        supervised_words = [tok._word(i) for i, l in zip(out["input_ids"], out["labels"])
                            if l != IGNORE_INDEX]
        assert "ARTIFACT" not in supervised_words
        assert supervised_words == [tok._word(i) for i in expected_supervised_words(tok)]

    def test_fallback_path_matches_offset_path(self):
        words = ALL_WORDS
        primary = tokenize_sft_messages(SAMPLE_MESSAGES, TinyTokenizer(extra_words=words))
        fallback = tokenize_sft_messages(SAMPLE_MESSAGES, NoOffsetTokenizer(extra_words=words))
        assert primary["input_ids"] == fallback["input_ids"]
        assert primary["labels"] == fallback["labels"]

    def test_broken_template_raises_on_fallback(self):
        with pytest.raises(SftSerializationError):
            tokenize_sft_messages(SAMPLE_MESSAGES, BrokenTemplateTokenizer())

    def test_must_end_with_assistant(self):
        tok = TinyTokenizer()
        with pytest.raises(SftSerializationError):
            tokenize_sft_messages(SAMPLE_MESSAGES[:-1], tok)

    def test_truncation_drops_not_cuts(self):
        tok = TinyTokenizer()
        assert tokenize_sft_messages(SAMPLE_MESSAGES, tok, max_seq_len=10**6) is not None
        assert tokenize_sft_messages(SAMPLE_MESSAGES, tok, max_seq_len=5) is None

    def test_chat_template_kwargs_passthrough_and_fallback(self):
        class RecordingTokenizer(TinyTokenizer):
            def __init__(self):
                super().__init__(extra_words=ALL_WORDS)
                self.seen_kwargs = []

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False, chat_template_kwargs=None):
                self.seen_kwargs.append(chat_template_kwargs)
                return super().apply_chat_template(messages, tokenize, add_generation_prompt,
                                                   chat_template_kwargs)

        tok = RecordingTokenizer()
        kwargs = {"enable_thinking": False}
        tokenize_sft_messages(SAMPLE_MESSAGES, tok, chat_template_kwargs=kwargs)
        assert any(k == kwargs for k in tok.seen_kwargs)

        class StrictTokenizer(TinyTokenizer):
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False, chat_template_kwargs=None):
                if chat_template_kwargs is not None:
                    raise TypeError("unexpected chat_template_kwargs")
                return super().apply_chat_template(messages, tokenize, add_generation_prompt,
                                                   None)

        out = tokenize_sft_messages(SAMPLE_MESSAGES, StrictTokenizer(extra_words=ALL_WORDS),
                                    chat_template_kwargs=kwargs)
        assert out is not None and len(out["input_ids"]) == len(out["labels"])


class TestRecordLevel:
    def _fake_trajectory(self, verified=True, clean=True, split="train"):
        from toolforge.trajectories import Trajectory
        return Trajectory(
            trajectory_id="traj-x", task_id="tf-cal-ss-001", split=split, family="calendar",
            category="single_step_selection", difficulty="easy", source="reference",
            policy_id="test", sampling_seed=0, request="create", messages=SAMPLE_MESSAGES,
            actions=[{"action": "tool_call"}, {"action": "final_answer"}],
            termination="final_answer", final_output="done", verified=verified,
            verification={"success": verified}, parse_error_count=0 if clean else 1,
            invalid_call_count=0 if clean else 1)

    def test_filters(self):
        assert trajectory_to_sft_record(self._fake_trajectory()) is not None
        assert trajectory_to_sft_record(self._fake_trajectory(verified=False)) is None
        assert trajectory_to_sft_record(self._fake_trajectory(clean=False)) is None
        assert trajectory_to_sft_record(self._fake_trajectory(clean=False),
                                        require_clean=False) is not None

    def test_build_records_stats(self):
        trajs = [self._fake_trajectory(), self._fake_trajectory(verified=False),
                 self._fake_trajectory(clean=False)]
        records, stats = build_sft_records(trajs)
        assert len(records) == 1
        assert stats == {"total": 3, "unverified": 1, "not_clean": 1, "kept": 1}

    def test_file_roundtrip_and_tokenize(self, tmp_path):
        records = [{"task_id": "t1", "trajectory_id": "a", "source": "reference",
                    "split": "train", "family": "calendar",
                    "category": "single_step_selection", "messages": SAMPLE_MESSAGES}]
        path = tmp_path / "sft.jsonl"
        save_sft_records(path, records)
        assert load_sft_records(path)[0]["messages"] == SAMPLE_MESSAGES
        tok = TinyTokenizer()
        dataset = tokenize_sft_file(path, tok, max_seq_len=1024)
        assert len(dataset) == 1 and dataset.dropped_too_long == 0

    def test_all_dropped_raises(self, tmp_path):
        records = [{"task_id": "t1", "messages": SAMPLE_MESSAGES}]
        path = tmp_path / "sft.jsonl"
        save_sft_records(path, records)
        with pytest.raises(SftSerializationError, match="exceeded max_seq_len"):
            tokenize_sft_file(path, TinyTokenizer(), max_seq_len=5)

    def test_load_rejects_corrupt(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"task_id": "x"}\n', encoding="utf-8")
        with pytest.raises(SftSerializationError):
            load_sft_records(path)
