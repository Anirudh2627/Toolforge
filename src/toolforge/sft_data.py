"""SFT sample construction: trajectory → messages → masked token ids.

Serialization contract (identical at training, inference, and evaluation):

- Conversations are the executor's messages (system with embedded tool
  catalog, user request, assistant JSON actions, tool observations), rendered
  with the tokenizer's chat template — the same rendering
  :class:`toolforge.inference.HuggingFacePolicy` uses.
- **Loss is computed on assistant spans only** (including the template's
  end-of-turn tokens); system/user/tool tokens get label ``-100``.
- Masking is derived by *token-level prefix differencing*: for each assistant
  turn we render the conversation up to that turn with and without the
  generation prompt; the difference is exactly the supervised span. If a chat
  template breaks the prefix-stability assumption, tokenization raises
  :class:`SftSerializationError` instead of silently mislabeling tokens.
- Samples longer than ``max_seq_len`` are **dropped and counted**, never
  silently truncated (truncation could cut the final assistant turn, which is
  the behavior we most need supervised).

This module is tokenizer-only (no torch needed for message-level records);
the token-id functions require a Hugging Face-compatible tokenizer object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from toolforge.schemas import ToolForgeError
from toolforge.trajectories import Trajectory

IGNORE_INDEX = -100


class SftSerializationError(ToolForgeError):
    """Raised when a chat template violates the prefix assumptions of masking."""


# --------------------------------------------------------------------------
# Message-level records (tokenizer-agnostic, inspectable, committed to disk)
# --------------------------------------------------------------------------

def trajectory_to_sft_record(traj: Trajectory, *, require_verified: bool = True,
                             require_clean: bool = True) -> dict[str, Any] | None:
    """Convert a trajectory into an SFT messages record, or None if filtered.

    Filters (documented, deliberate):
    - ``require_verified``: only verifier-passed trajectories teach task success.
    - ``require_clean``: trajectories containing parse failures or invalid tool
      calls are excluded — imitating recovered mistakes is a Phase 4 (RL) job,
      not an SFT one.
    """
    if require_verified and not traj.verified:
        return None
    if require_clean and (traj.parse_error_count > 0 or traj.invalid_call_count > 0):
        return None
    messages = traj.messages
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise SftSerializationError(f"Trajectory {traj.trajectory_id}: bad message structure")
    if messages[-1]["role"] != "assistant":
        raise SftSerializationError(
            f"Trajectory {traj.trajectory_id}: conversation must end with an assistant turn")
    return {
        "task_id": traj.task_id,
        "trajectory_id": traj.trajectory_id,
        "source": traj.source,
        "split": traj.split,
        "family": traj.family,
        "category": traj.category,
        "messages": messages,
    }


def build_sft_records(trajectories: Iterable[Trajectory], **filter_kwargs: Any) -> tuple[list[dict], dict[str, int]]:
    """Filter+convert a trajectory stream; returns (records, filter_stats)."""
    records: list[dict] = []
    stats = {"total": 0, "unverified": 0, "not_clean": 0, "kept": 0}
    for traj in trajectories:
        stats["total"] += 1
        if not traj.verified:
            stats["unverified"] += 1
        elif traj.parse_error_count > 0 or traj.invalid_call_count > 0:
            stats["not_clean"] += 1
        record = trajectory_to_sft_record(traj, **filter_kwargs)
        if record is not None:
            records.append(record)
            stats["kept"] += 1
    return records, stats


def save_sft_records(path: str | Path, records: list[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise SftSerializationError(f"SFT file not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rec = json.loads(line)
                    if not rec.get("messages"):
                        raise ValueError("record has no messages")
                    out.append(rec)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SftSerializationError(f"{path}:{line_no}: {exc}") from exc
    return out


# --------------------------------------------------------------------------
# Tokenization with assistant-only loss masks
# --------------------------------------------------------------------------

def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _render(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool,
            chat_template_kwargs: dict[str, Any] | None) -> str:
    """Render with optional template kwargs (e.g. Qwen3 ``enable_thinking``).

    Falls back to rendering without the kwargs when a template rejects them,
    mirroring :class:`toolforge.inference.HuggingFacePolicy` so training and
    inference serialization stay byte-identical for the same tokenizer.
    """
    if chat_template_kwargs:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs)
        except (TypeError, ValueError):
            pass  # template without kwargs support: identical fallback on both sides
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _char_to_token_labels(offsets: list[tuple[int, int]], n_tokens: int,
                          char_spans: list[tuple[int, int]]) -> list[int]:
    """Map supervised character spans to per-token labels (1 if overlapped)."""
    labels = [0] * n_tokens
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:  # special tokens with empty offsets stay masked
            continue
        for lo, hi in char_spans:
            if tok_start < hi and lo < tok_end:
                labels[tok_idx] = 1
                break
    return labels


def tokenize_sft_messages(messages: list[dict[str, str]], tokenizer: Any,
                          max_seq_len: int | None = None,
                          chat_template_kwargs: dict[str, Any] | None = None
                          ) -> dict[str, list[int]] | None:
    """Tokenize one conversation; labels supervise assistant output only.

    Masking algorithm (character-offset based; template-robust):

    1. Render the full conversation once -> the exact text trained on, tokenized
       with ``return_offsets_mapping`` so every token maps back to characters.
    2. For each assistant turn, locate ``message.content`` in the rendered text
       (sequentially, left to right), then extend the supervised span to the
       start of the next message block — which necessarily begins with the
       template's message-start marker, detected as the most frequent special
       marker in the render (e.g. Qwen's im-start token). This captures the
       content plus the template's end-of-turn tokens, and excludes tool
       observations, system/user text, and any artifacts the template injects
       only for last-position messages (verified against Qwen3's empty-think
       injection under ``enable_thinking=False``).
    3. Tokens overlapping a supervised character span get ``labels = input_ids``;
       everything else is ``-100``.

    Falls back to a strict token-prefix-diff method for tokenizers without
    offset mapping (test fakes), which requires the generation prompt to be a
    token prefix of the full render and raises otherwise.

    Returns ``{"input_ids", "labels", "attention_mask"}`` or ``None`` when the
    sample exceeds ``max_seq_len`` (dropped by policy, counted by the caller).
    Raises :class:`SftSerializationError` rather than producing a wrong mask.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise SftSerializationError("conversation must end with an assistant turn")
    if not any(m["role"] == "assistant" for m in messages):
        raise SftSerializationError("conversation contains no assistant turn")

    full_text = _render(tokenizer, messages, False, chat_template_kwargs)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    # BatchEncoding is a UserDict (Mapping, not dict) — duck-type the lookup.
    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    input_ids = list(encoded["input_ids"])

    if offsets is None:
        return _tokenize_by_prefix_diff(messages, tokenizer, full_text, input_ids,
                                        chat_template_kwargs, max_seq_len)

    offsets = [tuple(o) for o in offsets]
    if len(offsets) != len(input_ids):
        raise SftSerializationError("offset_mapping length mismatch; cannot mask safely")

    # Message-start marker: most frequent special-marker candidate in the render.
    candidates = [t for t in (getattr(tokenizer, "additional_special_tokens", None) or [])
                  if isinstance(t, str) and t in full_text]
    for guess in ("<|im_start|>", "<|startoftext|>", "<|user|>", "<|system|>"):
        if guess in full_text:
            candidates.append(guess)
    marker = max(candidates, key=full_text.count) if candidates else None

    char_spans: list[tuple[int, int]] = []
    search_from = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        pos = full_text.find(content, search_from)
        if pos < 0:
            raise SftSerializationError(
                f"Assistant content for turn {index} not found in the rendered "
                "conversation; the template may transform message content.")
        end = pos + len(content)
        if index + 1 < len(messages):
            if marker:
                nxt = full_text.find(marker, end)
                if nxt >= 0:
                    end = nxt
        else:
            # Final turn: supervise through the end of the render so the
            # template's closing end-of-turn tokens are always learned.
            end = len(full_text)
        char_spans.append((pos, end))
        search_from = end

    label_mask = _char_to_token_labels(offsets, len(input_ids), char_spans)
    labels = [tok if keep else IGNORE_INDEX for tok, keep in zip(input_ids, label_mask)]
    supervised = sum(label_mask)
    if supervised == 0:
        raise SftSerializationError("no supervised tokens after masking")

    if max_seq_len is not None and len(input_ids) > max_seq_len:
        return None
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def _tokenize_by_prefix_diff(messages: list[dict[str, str]], tokenizer: Any,
                             full_text: str, full_ids: list[int],
                             chat_template_kwargs: dict[str, Any] | None,
                             max_seq_len: int | None) -> dict[str, list[int]] | None:
    """Strict fallback for tokenizers without offset mapping (test fakes).

    Requires: render(prefix, gen=True) is a token prefix of the full render,
    and render(prefix + assistant) = render(prefix, gen=True) + span.
    """
    labels = [IGNORE_INDEX] * len(full_ids)
    assistant_positions = [k for k, m in enumerate(messages) if m["role"] == "assistant"]
    for i, k in enumerate(assistant_positions):
        prompt_ids = _encode(tokenizer, _render(tokenizer, messages[:k], True,
                                                chat_template_kwargs))
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise SftSerializationError(
                f"Chat template is not anchor-stable at assistant turn {k} and the "
                "tokenizer lacks offset_mapping; cannot mask safely.")
        turn_ids = _encode(tokenizer, _render(tokenizer, messages[:k + 1], False,
                                              chat_template_kwargs))
        if turn_ids[:len(prompt_ids)] != prompt_ids:
            raise SftSerializationError(
                f"Chat template violates the local prefix property at turn {k}")
        end = len(turn_ids)
        if i + 1 < len(assistant_positions):
            end = min(end, len(_encode(
                tokenizer, _render(tokenizer, messages[:assistant_positions[i + 1]], True,
                                   chat_template_kwargs))))
        end = max(end, len(prompt_ids) + 1)
        if full_ids[len(prompt_ids):end] != turn_ids[len(prompt_ids):end]:
            raise SftSerializationError(
                f"span mismatch against full render at turn {k}")
        labels[len(prompt_ids):end] = full_ids[len(prompt_ids):end]
    if not any(l != IGNORE_INDEX for l in labels):
        raise SftSerializationError("no supervised tokens after masking")
    if max_seq_len is not None and len(full_ids) > max_seq_len:
        return None
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


@dataclass
class TokenizedSftDataset:
    """In-memory tokenized dataset with drop statistics (torch-free until __getitem__)."""

    samples: list[dict[str, list[int]]]
    meta: list[dict[str, Any]]
    dropped_too_long: int = 0

    def __len__(self) -> int:
        return len(self.samples)


def tokenize_sft_file(path: str | Path, tokenizer: Any, max_seq_len: int = 1024,
                      chat_template_kwargs: dict[str, Any] | None = None) -> TokenizedSftDataset:
    records = load_sft_records(path)
    samples: list[dict[str, list[int]]] = []
    meta: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        tok = tokenize_sft_messages(rec["messages"], tokenizer, max_seq_len=max_seq_len,
                                    chat_template_kwargs=chat_template_kwargs)
        if tok is None:
            dropped += 1
            continue
        samples.append(tok)
        meta.append({k: rec.get(k) for k in ("task_id", "trajectory_id", "source", "family",
                                             "category")})
    if not samples:
        raise SftSerializationError(
            f"{path}: every sample exceeded max_seq_len={max_seq_len} (dropped={dropped}); "
            "raise the limit or fix the data")
    return TokenizedSftDataset(samples=samples, meta=meta, dropped_too_long=dropped)
