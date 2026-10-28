# ToolForge

**Execution-verified tool-use training and evaluation for small language models.**

ToolForge trains and evaluates a small open model (target: Qwen3-0.6B → 1.7B-class, QLoRA)
on tool-use tasks inside six deterministic, sandboxed mock environments. Task success is
computed from the **actual resulting environment state** — not from text similarity with a
reference answer — which makes the same verifier usable for benchmarking, rejection-sampled
SFT data, and GRPO rewards.

**Central research question:** *Can execution-verified SFT followed by GRPO improve the
task-completion accuracy of a small open model on tool-use tasks, including unseen tool
families, without an unacceptable increase in latency or inference cost?* The experiments
will determine the answer; nothing is assumed.

> **Status: Phase 1 complete** (foundation: environments, executor, verifier, benchmark,
> CPU tests). Phases 2–6 (evaluation harness, SFT, GRPO, ablations, Colab/demo) follow.
> **No results are reported anywhere in this repository until real runs exist.**

---

## Architecture

```mermaid
flowchart LR
    subgraph Benchmark
        TB[tasks_builders.py\n192 deterministic tasks\n12 categories, 6 families] --> TJ[benchmark_v1.jsonl\nfixture + expected outcome + split]
    end
    subgraph Episode
        P[Policy\nscripted / base / SFT / SFT+GRPO] -->|one JSON action per turn| EX[Executor\nvalidate → route → execute]
        EX -->|structured observation| P
        EX <--> ENV[6 mock environments\ncalendar · email · files\ncrm · sql · todo]
    end
    TJ --> EX
    EX -->|trace + state snapshots| V[Verifier\nstate checks, not text similarity]
    TJ -->|expected outcome| V
    V -->|success, failure categories| M[Metrics / plots / reports]
    V -.->|Phase 3–4: rejection sampling + reward| TR[SFT / GRPO training]
```

## Repository layout (Phase 1)

```text
ToolForge/
├── configs/base.yaml            # env/benchmark/paths settings (strict key validation)
├── src/toolforge/
│   ├── config.py                # dataclass configs, YAML loading, ConfigError on typos
│   ├── schemas.py               # action grammar, ToolSpec, JSON-Schema-subset validator
│   ├── timeutils.py             # date/time helpers, free-range computation (fixed simulated clock)
│   ├── registry.py              # EnvironmentRegistry + ToolRouter (multi-family episodes)
│   ├── executor.py              # episode loop: parse → validate → execute → trace, loop guard
│   ├── verifier.py              # declarative state checks, side-effect diffing, verdicts
│   ├── tasks.py                 # Task/ExpectedOutcome model, splits, leakage checks, JSONL I/O
│   ├── tasks_builders.py        # deterministic benchmark construction (templates + handwritten)
│   └── environments/            # base.py, calendar.py, email.py, files.py, crm.py, sql.py,
│                                # todo.py, pools.py, registry.py
├── scripts/generate_tasks.py    # regenerate data/tasks/benchmark_v1.jsonl
├── tests/                       # 248 CPU-only tests (no model downloads)
├── data/tasks/                  # generated benchmark (JSONL + summary)
└── configs, LICENSE (.env.example, requirements*.txt, pyproject.toml)
```

Deliberate deviations from the original tree: `tasks.py` + `tasks_builders.py` +
`verifier.py` + `timeutils.py` were added (benchmark content vs. plumbing separation),
task management lives in `environments/todo.py` (name collision with benchmark tasks),
and there is a single registry (`registry.py`) plus default-registration
(`environments/registry.py`). `rewards.py`, `metrics.py`, `inference.py`,
`data_generation.py`, `evaluation.py`, `checkpointing.py` and the MCP adapter arrive in
Phases 2–4; the MCP adapter will wrap these same `ToolSpec`s without the core depending
on any MCP package.

## Installation (CPU-only core)

```bash
git clone <this repo> && cd ToolForge
python -m pip install -r requirements-dev.txt   # PyYAML + pytest
# or: python -m pip install -e ".[dev]"
```

Python ≥ 3.10. The core has **no torch/transformers dependency** — environments,
executor, verifier, and benchmark generation run on any CPU (verified on 3.11).

## Tests

```bash
python -m pytest tests/ -q          # 248 tests, < 2 s, CPU-only, no downloads
```

Covered: schema validation and action parsing (incl. malformed-output categories),
environment determinism/isolation/reset, per-family state transitions, SQL safety
(authorizer denials, multi-statement rejection, runaway-CTE abort), executor loop
(step limit, loop guard, error feedback, context hygiene), verifier checks and
failure categories, benchmark determinism, split integrity, and leakage prevention.

## Benchmark

```bash
python scripts/generate_tasks.py --config configs/base.yaml
# -> data/tasks/benchmark_v1.jsonl (192 tasks) + benchmark_v1_summary.json
```

| Split | Tasks | Notes |
|---|---|---|
| train | 116 | excludes the held-out family entirely |
| val | 17 | hyperparameter selection only |
| test | 59 | includes all 31 `crm` (held-out family) tasks |

- **12 categories**: single-step selection (31), argument construction (25), multi-step
  workflows (28), retrieve-then-act (28), dependent actions (15), ambiguous→clarify (13),
  missing-info→no-action (9), invalid-tool bait (12), indirect reference (10), error
  recovery (11), conflicting constraints (4), impossible tasks (6).
- **6 families**: calendar (38), email (33), files (30), crm (31 — *held out from
  training*), sql (27), todo (33).
- Splits are assigned by a stable hash of `(salt, task_id)`; held-out families go
  entirely to test; every category is guaranteed ≥1 test task; `check_no_leakage` fails
  loudly on any violation. Building twice is byte-identical (tested).
- Every task embeds its **full explicit fixture**, so expected outcomes can never drift
  when generators change.

### Task record (JSONL, abbreviated)

```json
{"task_id": "tf-cal-ss-001", "family": "calendar", "category": "single_step_selection",
 "request": "Create a calendar event titled 'Project review' on 2026-03-04 from 13:00 to 14:00.",
 "fixture": {"current_date": "2026-03-02", "events": [...], ...},
 "tool_families": ["calendar"], "max_steps": 3, "split": "train",
 "expected": {"terminal_action": "final_answer", "forbid_state_changes": false,
   "checks": [{"check": "record_exists", "collection": "events",
               "where": {"title": "Project review", "date": "2026-03-04",
                          "start": "13:00", "end": "14:00", "status": "scheduled"}}]}}
```

## Action grammar (frozen — used identically in training, inference, and evaluation)

The model emits exactly one JSON object per turn:

```json
{"action": "tool_call", "tool": "calendar.create_event", "arguments": {"title": "...", "date": "2026-03-04", "start": "13:00", "end": "14:00"}}
{"action": "clarify", "question": "Which day works for you?"}
{"action": "final_answer", "content": "Done — created the event."}
```

Parsing tolerates prose around the JSON and ```` ```json ```` fences, but is strict about
structure. Failures are categorized (`invalid_json`, `unknown_action`, `missing_field`,
`field_type_error`, `unexpected_field`, `tool_name_format`) and fed back to the model as
structured error observations; they count toward the step budget and the invalid-call
metrics.

## Environments and verifier

Six deterministic mock environments with in-memory state, a fixed simulated clock
(`current_date = 2026-03-02`, never the system time), isolated per-episode instances,
JSON-serializable snapshots, and structured tool errors (`not_found`,
`invalid_argument`, `already_exists`, `ambiguous`, `unsafe_operation`, `unsupported`,
`limit_exceeded`, `unknown_tool`, `internal_error`). The SQL family enforces read-only
execution with a SQLite authorizer (writes/DDL/ATTACH/PRAGMA denied), single-statement
queries, a VM-step cap against runaway recursion, and row truncation.

The **verifier** decides success from:
1. terminal action (final_answer vs clarify vs none) against the expected behavior,
2. a forbidden-side-effect diff between initial and final state snapshots (ignoring each
   family's declared volatile keys, e.g. email `read` flags),
3. declarative checks — `record_exists/absent`, `record_count`, `field_equals`,
   `no_overlaps`, `tool_called(+arguments_match)`, `tool_not_called`,
   `answer_contains_all/any`, `invalid_calls_at_most` — where ground truth was **computed
   from the fixture at build time**,
4. step-limit / loop-guard termination.

Anti-reward-hacking properties (all tested): claiming success without acting fails;
acting when forbidden fails even with a correct answer; a no-op episode passes **zero**
of the 192 tasks; no-action tasks must also surface the missing information; the
model-visible context provably never contains expected outcomes; repeated identical
calls trigger the loop guard instead of accumulating reward. Lexical answer checks are
used only where state cannot express correctness (information retrieval, refusals) and
are documented as such.

## Results

| Model | Overall success | Seen families | Held-out (crm) | Invalid-call rate | Latency/task |
|---|---|---|---|---|---|
| Qwen3-0.6B base | *pending Phase 2* | *pending* | *pending* | *pending* | *pending* |
| + SFT (QLoRA) | *pending Phase 3* | *pending* | *pending* | *pending* | *pending* |
| + SFT + GRPO | *pending Phase 4* | *pending* | *pending* | *pending* | *pending* |

No cell in this table will be filled from anything other than saved evaluation runs.

## Roadmap

- **Phase 2** — evaluation harness: model-agnostic inference wrapper, metrics + bootstrap
  CIs, per-task prediction/traces JSONL, `scripts/evaluate.py`, `scripts/benchmark_inference.py`,
  MCP adapter (optional import).
- **Phase 3** — trajectories + SFT: reference/self-generated trajectories with rejection
  sampling, chat-template serialization (loss masked on tool observations), QLoRA training,
  pinned torch/transformers/peft/trl versions (`requirements-train.txt`).
- **Phase 4** — GRPO on top of the SFT policy using the verifier-backed reward.
- **Phase 5** — ablations, plots from real evaluation files only, latency/cost benchmarking.
- **Phase 6** — Colab notebook, Streamlit demo, full report.

## Known limitations (Phase 1)

- **Lexical answer checks**: retrieval/refusal tasks verify computed ground-truth values
  by case-insensitive substring containment; a model can theoretically pass by mentioning
  a value without a correct sentence (mitigated by requesting exact values; state checks
  dominate elsewhere). Count answers can substring-collide with dates/ids (e.g. `3` in
  `2026-03-03`) — requests instruct "answer with the exact number", and Phase 2 metrics
  will report per-category results so any distortion is visible.
- **Single-turn JSON grammar**: no streaming/partial JSON; a turn that emits two JSON
  objects uses the first one.
- **Held-out family = crm only** (configurable): "unseen combinations of known tools"
  (multi-family tasks) are supported by the executor/router but not yet populated in v1
  of the benchmark; they arrive with Phase 2 evaluation slices.
- **Simulated clock**: all fixtures use a fixed `current_date`; relative-time reasoning
  ("tomorrow") is resolved at build time in requests/expectations.
- **SQL dialect**: SQLite; recursive CTEs are allowed but step-capped. The authorizer
  denies `PRAGMA`, so schema discovery goes through `list_tables`/`describe_table`.

## License

Code and generated benchmark: MIT (see `LICENSE`). Base model weights are **not**
redistributed and remain under their upstream licenses; adapter publishing (Phase 6)
will document the base-model terms separately. All data is synthetic — no real emails,
files, contacts, or database contents.
