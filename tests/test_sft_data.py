"""Tests for SFT serialization: assistant-only loss masks, truncation policy,
record filtering. Uses a deterministic whitespace tokenizer with a ChatML-like
template, so masks can be asserted token-exactly without model downloads.
"""

from __future__ import annotations

import json

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

SPECIALS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<|unk|>"]
ROLES = ["system", "user", "assistant", "tool"]


class TinyTokenizer:
    """Whitespace tokenizer with a ChatML-like template (prefix-stable)."""

    def __init__(self, extra_words=()):
        self.vocab = {w: i for i, w in enumerate(SPECIALS + ROLES)}
        for word in extra_words:
            self._add(word)
        self.pad_token_id = 0
        self.eos_token_id = self.vocab["<|im_end|>"]
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"

    def _add(self, word: str) -> None:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)

    def _ids(self, text: str) -> list[int]:
        unk = self.vocab["<|unk|>"]
        return [self.vocab.get(w, unk) for w in text.split()]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": self._ids(text)}

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        parts = [f"<|im_start|> {m['role']} {m['content']} <|im_end|>" for m in messages]
        text = " ".join(parts)
        if add_generation_prompt:
            text += " <|im_start|> assistant"
        return text


class BrokenTemplateTokenizer(TinyTokenizer):
    """Violates the generation-prompt prefix property on purpose."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            chat_template_kwargs=None):
        text = super().apply_chat_template(messages, tokenize, False, chat_template_kwargs)
        if add_generation_prompt:
            text += " ASSISTANT-DIFFERENT-MARKER"
        return text


SAMPLE_MESSAGES = [
    {"role": "system", "content": "you are toolforge"},
    {"role": "user", "content": "create event sync tomorrow"},
    {"role": "assistant", "content": "calling calendar create now"},
    {"role": "tool", "content": "ok true event created"},
    {"role": "assistant", "content": "done created the event"},
]


def expected_supervised_words(tokenizer: TinyTokenizer) -> list[int]:
    ids = []
    for m in SAMPLE_MESSAGES:
        if m["role"] == "assistant":
            ids.extend(tokenizer._ids(m["content"]))
            ids.append(tokenizer.vocab["<|im_end|>"])  # end-of-turn must be supervised
    return ids


class TestMasking:
    def test_labels_cover_exactly_assistant_spans(self):
        tok = TinyTokenizer(extra_words="you are toolforge create event sync tomorrow "
                                        "calling calendar now ok true created done the".split())
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        supervised = [i for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX]
        assert supervised == expected_supervised_words(tok)
        # supervised labels equal the input tokens at those positions
        assert all(i == l for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX)

    def test_non_assistant_tokens_are_masked(self):
        # Full vocab so no word collides with <unk> inside supervised spans.
        tok = TinyTokenizer(extra_words=(
            "you are toolforge create event sync tomorrow calling calendar now "
            "ok true created done the").split())
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        masked_positions = sum(1 for l in out["labels"] if l == IGNORE_INDEX)
        assert masked_positions > 0
        # system/user/tool words appear only in masked positions
        for word in ("toolforge", "tomorrow", "true"):
            wid = tok.vocab.get(word, tok.vocab["<|unk|>"])
            positions = [k for k, i in enumerate(out["input_ids"]) if i == wid]
            assert all(out["labels"][k] == IGNORE_INDEX for k in positions), word

    def test_input_length_matches_rendered_conversation(self):
        tok = TinyTokenizer()
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        rendered = tok.apply_chat_template(SAMPLE_MESSAGES, tokenize=False)
        assert len(out["input_ids"]) == len(rendered.split())
        assert out["attention_mask"] == [1] * len(out["input_ids"])

    def test_end_of_turn_token_supervised(self):
        tok = TinyTokenizer()
        out = tokenize_sft_messages(SAMPLE_MESSAGES, tok)
        eot = tok.vocab["<|im_end|>"]
        # last token of the sequence is the final assistant turn's end-of-turn
        assert out["input_ids"][-1] == eot
        assert out["labels"][-1] == eot

    def test_broken_template_raises(self):
        tok = BrokenTemplateTokenizer()
        with pytest.raises(SftSerializationError):
            tokenize_sft_messages(SAMPLE_MESSAGES, tok)

    def test_must_end_with_assistant(self):
        tok = TinyTokenizer()
        with pytest.raises(SftSerializationError):
            tokenize_sft_messages(SAMPLE_MESSAGES[:-1], tok)

    def test_truncation_drops_not_cuts(self):
        tok = TinyTokenizer()
        assert tokenize_sft_messages(SAMPLE_MESSAGES, tok, max_seq_len=10**6) is not None
        assert tokenize_sft_messages(SAMPLE_MESSAGES, tok, max_seq_len=5) is None


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
        # explicit opt-out keeps dirty trajectories
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
        path.write_text('{"task_id": "x"}\n', encoding="utf-8")  # no messages
        with pytest.raises(SftSerializationError):
            load_sft_records(path)
