# ToolForge Research Report — Outline

This outline structures the final report (Phase 6 deliverable per the build
spec). Every empirical section must be filled **only** from saved artifacts:
`outputs/eval/*/metrics.json` + `predictions.jsonl`, `outputs/checkpoints/*/
run_metadata.json`, `outputs/checkpoints/grpo/history.jsonl`,
`reports/ablations/REPORT.md`, and `outputs/benchmark*.json`. If a run does not
exist, the section says so — it is never filled with expected or placeholder
numbers.

---

## 1. Motivation & research question

- Tool-use reliability of small open models; why execution-grounded signals
  (verifiable rewards) matter more for agents than for chat.
- **RQ:** Can execution-verified SFT followed by GRPO improve task-completion
  accuracy of a small open model on tool-use tasks — including unseen tool
  families — without unacceptable latency/inference-cost increase?
- State the answer as *found*, with CIs, whatever it is. Negative results are
  reportable results.

## 2. Why final-state verification

- Text-similarity scoring rewards fluent failure and punishes valid alternative
  solutions; state-based verification accepts any action sequence that produces
  the required final state and rejects claims without effects (measured: a
  confident "I did it" no-op scores 0 on all 192 tasks).
- The verifier as a *reward oracle*: identical code path for benchmark scoring,
  SFT rejection sampling, and GRPO rewards — no train/eval objective drift.
- Anti-hacking properties and their tests (forbidden side effects, volatile-key
  semantics, loop guard, invalid-call budgets, no-answer-without-evidence in
  reference data).

## 3. Benchmark design

- 192 tasks / 12 categories / 6 families; distribution rationale (why
  single-step and multi-step dominate; why impossible/bait are rare).
- Deterministic construction: seeded fixtures, build-time ground truth,
  byte-identical rebuilds (md5), self-contained records.
- Splits: stable-hash assignment, held-out family (`crm`, all 31 tasks in
  test), category-coverage balancing, leakage tests.
- Threats: lexical answer checks for retrieval/refusal categories (substring
  collisions), single held-out family, simulated-clock scope, template-task
  family resemblance (models may exploit surface patterns — mitigated by
  held-out family + error/bait categories, not eliminated).

## 4. SFT data pipeline

- Reference-solution compiler: spec → actions; evidence-first rule (retrieval
  answers must be observed in a tool result before being stated); `$prev`
  runtime-id resolution; 133/133 verified coverage; what "compiled" means for
  data diversity (terse style; rollouts/teachers as the diversity mechanism).
- Rejection sampling results: acceptance rates per source (reference vs base
  model vs teacher, as measured), rejection category distributions.
- Serialization: chat-template rendering, assistant-only offset masking
  (validated on the real Qwen3 template including the empty-think artifact),
  compact catalog decision + measured token savings, drop-not-truncate policy
  and measured drop rates.

## 5. RL objective and reward

- GRPO formulation used (group-relative advantages, clipped token-level
  surrogate, KL-to-adapter-disabled-reference); why not TRL (evidence: native
  tool-call dispatch vs content-JSON grammar; stateless reward protocol vs
  environment-state verification).
- Reward component table with the dominance guarantee (config-level rejection
  of shaping-dominant weights); measured component distributions from
  `history.jsonl`.
- Rollout mechanics: G per task, zero-variance skipping (fraction logged),
  CPU-env/GPU-model separation, measured seconds/iteration and budget model.

## 6. Evaluation protocol

- Identical tasks/tools/limits/initial states/verifier across models;
  fingerprints + grammar version recorded per run; metrics re-verified against
  predictions at comparison time.
- Metric definitions (denominators!), CI vocabulary: Wilson/bootstrap-over-
  tasks vs repeated-training variability — when each is available.
- Paired analysis + exact McNemar; why n≈59 test tasks limits significance.
- Latency/cost protocol: warm-up, medians/p95, policy-vs-env separation,
  cost only from measured time × user-supplied rate (labeled estimate).

## 7. Results (fill from artifacts only)

7.1 Headline: base vs SFT vs SFT+GRPO (success + CIs; distinguishable or not).
7.2 By category and by family (where gains/regressions concentrate).
7.3 Held-out family generalization (the RQ's transfer clause).
7.4 Behavior rates: invalid JSON/tool calls, argument failures, step limits,
    side effects, clarification P/R/F1.
7.5 Multi-step and dependent-action performance.
7.6 Latency & cost (hardware stated; estimate basis stated).
7.7 RL dynamics: training reward vs validated accuracy; divergence warnings.
7.8 Ablations actually run (data amounts, reward variants) — list what exists,
    mark the rest as not-run rather than padding.

## 8. Failure-mode analysis

- From `failure_category_counts` and per-task predictions: concrete examples
  (task ids), grouped: wrong tool selection, argument construction, premature
  final answer, over-clarification, loop behavior, unsafe-SQL handling.
- Reward-hacking attempts observed (or their absence) with evidence.

## 9. Limitations & threats to validity

- Single base-model family (Qwen3); single benchmark version; mock environments
  (no network/real APIs); lexical components of verification; Colab-class
  hardware limits (batch=1, no vLLM); smoke-vs-full config drift risk;
  run-to-run training variability only if repeat-groups were run.

## 10. Reproducibility statement

- Exact commands per stage (from notebook/CLI history), config snapshots and
  data fingerprints (in every `run_metadata.json`), seeds, library versions,
  hardware description, and the artifact tree needed to regenerate every
  figure and number: `compare_models.py` output + `reaggregate()` from
  predictions.

## 11. Next experiments

- Derived from Sections 7–9 (e.g., if held-out gap is large: cross-family
  training mix; if invalid-JSON rate persists: grammar-focused data; if reward
  diverges from accuracy: component re-weighting with the dominance checker;
  larger G / more iterations under a bigger budget; teacher-distillation arm).

---

*Authorship note: the report generator (`compare_models.py`) deliberately does
not write interpretive prose. Sections 1, 8 (grouping), 9, and 11 are author
text; every number anywhere must trace to an artifact path listed above.*
