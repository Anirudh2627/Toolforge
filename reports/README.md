# reports/

Experiment reports and plots generated **from actual evaluation outputs**
(Phase 5 tooling). No results are committed until real runs exist; scripts
refuse to plot missing data rather than inventing numbers.

- `REPORT_OUTLINE.md` — the research-report skeleton (what goes where, and the
  rule that every number traces to a saved artifact).
- After real runs: `ablations/` — `REPORT.md`, `comparison.json/csv`,
  `paired.json`, `provenance.json`, `plots/*.png`, produced by
  `scripts/compare_models.py` from evaluation run directories.

The illustrative preview mode (`compare_models.py --illustrative`) writes
watermarked fake-data figures for layout purposes only; those files are
prefixed `illustrative_` and must never be presented as results.
