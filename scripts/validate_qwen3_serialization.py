"""Validate SFT serialization against a REAL HF chat template (robust masking).

Run: PYTHONPATH=src python3 scripts/validate_qwen3_serialization.py [model_id]
Needs network once (tokenizer files only). Development-time validation utility;
the offline test suite covers the same logic via the TinyTokenizer fake.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toolforge.sft_data import IGNORE_INDEX, load_sft_records, tokenize_sft_messages

IM_END = "<" + "|im_end|>"
IM_START = "<" + "|im_start|>"
TOOL_RESP = "<" + "|tool_response|>"
THINK_OPEN = "<" + "think" + ">"


def main() -> int:
    from transformers import AutoTokenizer

    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id)
    records = load_sft_records(REPO_ROOT / "data" / "sft" / "sft_train.jsonl")
    kwargs = {"enable_thinking": False}

    sample = records[0]
    out = tokenize_sft_messages(sample["messages"], tok, max_seq_len=4096,
                                chat_template_kwargs=kwargs)
    n_sup = sum(1 for label in out["labels"] if label != IGNORE_INDEX)
    sup_ids = [i for i, l in zip(out["input_ids"], out["labels"]) if l != IGNORE_INDEX]
    decoded = tok.decode(sup_ids)
    print(f"{sample['task_id']}: {len(out['input_ids'])} tokens, "
          f"{n_sup} supervised ({n_sup / len(out['input_ids']):.0%})")
    print("supervised text:", repr(decoded[:300]))
    assert '"action": "tool_call"' in decoded and '"action": "final_answer"' in decoded
    for marker in (THINK_OPEN, IM_START, TOOL_RESP, "system"):
        assert marker not in decoded, f"non-assistant token supervised: {marker}"
    print("only assistant JSON turns are supervised  OK")

    msgs = sample["messages"]
    for k in range(2, len(msgs)):
        if msgs[k]["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:k], tokenize=False,
                                         add_generation_prompt=True,
                                         chat_template_kwargs=kwargs)
        full = tok.apply_chat_template(msgs[:k + 1], tokenize=False,
                                       add_generation_prompt=False,
                                       chat_template_kwargs=kwargs)
        assert full.startswith(prompt), f"local prefix property broken at turn {k}"
    print("local prefix property holds on the real template  OK")

    lens, over1024, over1536, supervised_fracs = [], 0, 0, []
    for r in records:
        o = tokenize_sft_messages(r["messages"], tok, chat_template_kwargs=kwargs)
        lens.append(len(o["input_ids"]))
        if len(o["input_ids"]) > 1024:
            over1024 += 1
        if len(o["input_ids"]) > 1536:
            over1536 += 1
        supervised_fracs.append(sum(1 for l in o["labels"] if l != IGNORE_INDEX)
                                / len(o["input_ids"]))
    lens_sorted = sorted(lens)
    print(f"token lengths: min={min(lens)} median={statistics.median(lens):.0f} "
          f"p95={lens_sorted[int(len(lens) * 0.95)]} max={max(lens)}")
    print(f"over 1024: {over1024}/{len(records)} ({over1024 / len(records):.1%}) | "
          f"over 1536: {over1536}/{len(records)} ({over1536 / len(records):.1%})")
    print(f"mean supervised fraction: {statistics.mean(supervised_fracs):.1%}")
    print("\nSERIALIZATION VALIDATION PASSED for", model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
