# ToolForge demo (Streamlit)

A lightweight UI over the **exact same** executor, verifier, and policies used
by the benchmark — nothing is re-implemented here.

```bash
pip install -r requirements-demo.txt     # streamlit (+ core requirements)
streamlit run demo/app.py
```

## What it does

- **Playground**: free-form request against any of the six simulated
  environments. Freeform runs have no ground truth, so the demo checks *clean
  completion only* and says so explicitly.
- **Benchmark tasks**: pick a real task from `data/tasks/benchmark_v1.jsonl`
  and run it fully verified — success is computed from the final environment
  state, with per-check pass/fail details, state diffs, failure-category
  explanations, and the verifier-backed reward breakdown.
- **Compare checkpoints**: run the same task with the compiled reference
  policy, the random baseline, and/or any loaded HF checkpoint (base, SFT,
  GRPO) side by side.
- **Tools & grammar**: browse every tool schema and the action grammar; the
  internal-executor vs MCP-adapter distinction is explained in-app.

## Safety properties (by construction)

- **No startup downloads.** The UI imports only the CPU core. HF weights load
  exclusively behind an explicit button that states the download size first,
  and are cached per session (`st.cache` via `st.session_state`).
- **No secrets, ever.** The demo never reads tokens or credentials; HF model
  ids and local adapter paths are typed by the user in the sidebar.
- **Everything is simulated.** All six environments are in-memory; no real
  emails, files, databases, or network side effects. A permanent banner says so.
- **Sensible fallbacks.** With no checkpoints trained yet, every tab works
  with the reference/random/no-op policies; missing benchmark files produce an
  actionable error message, not a crash.

## Architecture note

`demo/core.py` holds all logic (policy building, episode driving, verdict
explanation) with **no Streamlit imports**, so it is unit-tested headlessly in
`tests/test_demo_core.py`. `demo/app.py` is presentation only.
