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


def tokenize_sft_messages(messages: list[dict[str, str]], tokenizer: Any,
                          max_seq_len: int | None = None) -> dict[str, list[int]] | None:
    """Tokenize one conversation; labels supervise assistant spans only.

    Returns ``{"input_ids", "labels", "attention_mask"}`` or ``None`` when the
    sample exceeds ``max_seq_len`` (dropped by policy, counted by the caller).
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise SftSerializationError("conversation must end with an assistant turn")

    ids: list[int] = []
    labels: list[int] = []
    prefix: list[dict[str, str]] = []
    saw_assistant = False

    for message in messages:
        prefix.append(message)
        if message["role"] != "assistant":
            continue
        saw_assistant = True
        prompt_text = tokenizer.apply_chat_template(
            prefix[:-1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=False)
        prompt_ids = _encode(tokenizer, prompt_text)
        full_ids = _encode(tokenizer, full_text)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise SftSerializationError(
                "Chat template violates the generation-prompt prefix property; "
                "assistant spans cannot be located safely. Refusing to guess a mask.")
        if ids and prompt_ids[:len(ids)] != ids:
            raise SftSerializationError(
                "Chat template is not prefix-stable across turns; masking would be wrong.")
        if len(prompt_ids) < len(ids):
            raise SftSerializationError(
                "Rendered prompt shorter than accumulated tokens; template inconsistency.")
        # Mask everything up to the assistant span; supervise the span itself
        # (which includes the template's end-of-turn tokens).
        labels.extend([IGNORE_INDEX] * (len(prompt_ids) - len(ids)))
        assistant_span = full_ids[len(prompt_ids):]
        if not assistant_span:
            raise SftSerializationError("Empty assistant span; template renders nothing to learn")
        ids = prompt_ids + assistant_span
        labels.extend(assistant_span)

    if not saw_assistant:
        raise SftSerializationError("conversation contains no assistant turn")
    if max_seq_len is not None and len(ids) > max_seq_len:
        return None
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


@dataclass
class TokenizedSftDataset:
    """In-memory tokenized dataset with drop statistics (torch-free until __getitem__)."""

    samples: list[dict[str, list[int]]]
    meta: list[dict[str, Any]]
    dropped_too_long: int = 0

    def __len__(self) -> int:
        return len(self.samples)


def tokenize_sft_file(path: str | Path, tokenizer: Any, max_seq_len: int = 1024) -> TokenizedSftDataset:
    records = load_sft_records(path)
    samples: list[dict[str, list[int]]] = []
    meta: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        tok = tokenize_sft_messages(rec["messages"], tokenizer, max_seq_len=max_seq_len)
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
