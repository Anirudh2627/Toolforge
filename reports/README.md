# reports/

Experiment reports and plots generated **from actual evaluation outputs**
(Phase 5). No results are committed until real runs exist; scripts refuse to
plot missing data rather than inventing numbers.

Expected contents after the full pipeline has run:

- `ablations/` — comparison plots (base vs SFT vs SFT+GRPO) produced by
  `scripts/compare_models.py` from evaluation JSONL files.
- `REPORT.md` — concise write-up generated from aggregate metrics: strongest
  result, regressions, failure modes, limitations, next experiments.
