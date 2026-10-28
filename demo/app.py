"""ToolForge demo — Streamlit UI over demo/core.py.

Run:  streamlit run demo/app.py

Safety/UX rules honored here:
- NOTHING is downloaded at startup. HF checkpoints load only behind an
  explicit button that states the cost first.
- No secrets: the demo never reads tokens or env credentials; HF loading uses
  only public model ids / local adapter paths you type in.
- Everything is simulated in-memory; the banner says so permanently.
- Sensible fallbacks: with no checkpoints available the demo runs the
  compiled reference policy or the random baseline on real benchmark tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demo"))

import core  # noqa: E402  (demo/core.py)

st.set_page_config(page_title="ToolForge demo", page_icon="🛠️", layout="wide")

st.title("🛠️ ToolForge — tool-use agents, verified by execution")
st.caption("All operations are SIMULATED in-memory (no real email, files, databases, or "
           "network side effects). No secrets are used or requested. Success is computed "
           "from the actual final environment state — never from text similarity.")

TASKS_FILE = str(REPO_ROOT / core.DEFAULT_TASKS_FILE)


# --------------------------------------------------------------------------
# Sidebar: policy configuration (lazy model loading only)
# --------------------------------------------------------------------------

def load_hf_model(model_id: str):
    """Cached, explicit-load-only model fetch. Downloads weights (~1.2 GB for 0.6B)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


with st.sidebar:
    st.header("Policy")
    mode = st.radio("Run with", ["reference", "random", "hf", "scripted-final"],
                    format_func=lambda m: {
                        "reference": "Reference solution (compiled, CPU)",
                        "random": "Random baseline (CPU)",
                        "hf": "Hugging Face model / checkpoint",
                        "scripted-final": "No-op policy (immediate answer)",
                    }[m])
    seed = st.number_input("Seed", value=0, step=1)
    model = tokenizer = None
    model_desc = ""
    if mode == "hf":
        model_id = st.text_input("Base model id", value="Qwen/Qwen3-0.6B")
        adapter_id = st.text_input("Adapter dir (SFT/GRPO checkpoint, optional)", value="")
        compact = st.checkbox("Compact tool catalog (match how the checkpoint was trained)",
                              value=True)
        st.warning(f"Loading will download/allocate `{model_id}` weights "
                   "(~1.2 GB for 0.6B). Nothing loads until you press the button.")
        if st.button("Load model now", type="primary"):
            with st.spinner("Loading model (one time; cached for this session)..."):
                model, tokenizer = load_hf_model(model_id)
            st.session_state["tf_model"] = model
            st.session_state["tf_tokenizer"] = tokenizer
            st.session_state["tf_model_id"] = model_id
            st.session_state["tf_adapter_id"] = adapter_id or None
            st.session_state["tf_compact"] = compact
        model = st.session_state.get("tf_model")
        tokenizer = st.session_state.get("tf_tokenizer")
        adapter_id = st.session_state.get("tf_adapter_id", adapter_id or None)
        compact = st.session_state.get("tf_compact", compact)
        if model is None:
            st.info("No model loaded yet — the demo will fall back to the random baseline.")
    st.divider()
    st.markdown(render_grammar_help := core.render_policy_context_note())


def get_policy(task=None):
    """Build the configured policy, with a loud fallback when HF isn't loaded."""
    if mode == "hf":
        m = st.session_state.get("tf_model")
        tk = st.session_state.get("tf_tokenizer")
        if m is None or tk is None:
            st.warning("HF model not loaded — falling back to the random baseline for this run.")
            return core.build_policy("random", seed=int(seed))
        return core.build_policy("hf", model=m, tokenizer=tk,
                                 model_id=st.session_state.get("tf_model_id"),
                                 adapter_id=st.session_state.get("tf_adapter_id"),
                                 compact_catalog=bool(st.session_state.get("tf_compact", True)))
    return core.build_policy(mode, seed=int(seed), task=task)


def render_outcome(outcome: core.DemoOutcome, key_prefix: str = "") -> None:
    badge = "✅ verified success" if outcome.success else "❌ failed verification"
    st.markdown(f"**{badge}** · termination `{outcome.termination}` · "
                f"policy {outcome.policy_seconds:.2f}s / env {outcome.env_seconds:.3f}s")
    st.markdown("#### Tool-call trace")
    if not outcome.steps:
        st.write("_(no turns recorded)_")
    for s in outcome.steps:
        label = f"Turn {s.step}"
        if s.parse_error:
            label += f" — ⚠️ parse error ({s.parse_error['category']})"
        elif s.tool:
            ok = bool((s.result or {}).get("ok"))
            label += f" — `{s.tool}` {'✓' if ok else '✗ ' + str((s.result or {}).get('error', {}).get('code'))}"
        else:
            action = (s.action or {}).get("action", "?")
            label += f" — {action}"
        with st.expander(label, expanded=not bool(s.parse_error) and s.step == len(outcome.steps) - 1):
            if s.action:
                st.json(s.action)
            elif s.parse_error:
                st.json(s.parse_error)
            if s.result is not None:
                st.markdown("**observation**")
                st.json(s.result)
    if outcome.final_output:
        st.markdown("#### Final output")
        st.write(outcome.final_output)
    st.markdown("#### Why it passed / failed")
    for line in core.explain_outcome(outcome):
        st.write(line)
    if outcome.check_lines:
        with st.expander(f"Verifier checks ({len(outcome.check_lines)})"):
            for line in outcome.check_lines:
                st.code(line, language=None)
    if outcome.reward:
        with st.expander("Reward breakdown (verifier-backed)"):
            st.json(outcome.reward)
    with st.expander("Raw outcome JSON"):
        st.json(core.outcome_to_dict(outcome))


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_play, tab_bench, tab_compare, tab_tools = st.tabs(
    ["🕹️ Playground", "🎯 Benchmark tasks", "🆚 Compare checkpoints", "🧰 Tools & grammar"])

with tab_play:
    st.subheader("Freeform request (simulated environment)")
    st.caption("Freeform mode has no ground truth: it checks clean completion only. "
               "Use the Benchmark tab for real verification.")
    family = st.selectbox("Environment family", core.available_families(), key="play_family")
    request = st.text_area("Request", value="Create an event titled 'Demo sync' on 2026-03-04 "
                                            "from 10:00 to 11:00.", key="play_request")
    if st.button("Run request", type="primary", key="play_run"):
        task = core.make_freeform_task(request, family, seed=int(seed))
        if mode == "reference":
            st.info("Reference solutions exist only for benchmark tasks — "
                    "using the random baseline for this freeform run.")
            policy, desc = core.build_policy("random", seed=int(seed))
        else:
            policy, desc = get_policy(task=None)
        st.info(f"Policy: {desc}")
        with st.spinner("Running episode..."):
            outcome = core.run_demo(task, policy)
        render_outcome(outcome)

with tab_bench:
    st.subheader("Run a real benchmark task (fully verified)")
    fam_filter = st.selectbox("Family", [None] + core.available_families(), key="bench_family",
                              format_func=lambda f: f or "all")
    entries = core.list_benchmark_tasks(TASKS_FILE, family=fam_filter, limit=60)
    if not Path(TASKS_FILE).exists():
        st.error(f"Benchmark not found at {TASKS_FILE}. Run: "
                 "python scripts/generate_tasks.py --config configs/base.yaml")
    else:
        choice = st.selectbox(
            "Task", entries,
            format_func=lambda e: f"[{e['category']}/{e['difficulty']}] {e['task_id']}: "
                                  f"{e['request'][:70]}...", key="bench_choice")
        st.write(f"**Request:** {choice['request']}")
        st.write(f"Split: `{choice['split']}` — running a task here does not train anything; "
                 "the demo never modifies artifacts.")
        if st.button("Run task", type="primary", key="bench_run"):
            task = core.load_benchmark_task(choice["task_id"], TASKS_FILE)
            policy, desc = get_policy(task=task)
            st.info(f"Policy: {desc}")
            with st.spinner("Running episode..."):
                outcome = core.run_demo(task, policy)
            render_outcome(outcome)

with tab_compare:
    st.subheader("Compare policies on one task")
    st.caption("Base vs SFT vs GRPO checkpoints — when available. Each model loads only "
               "when you run this tab, one at a time, and shares the session cache.")
    entries_cmp = core.list_benchmark_tasks(TASKS_FILE, limit=60) if Path(TASKS_FILE).exists() else []
    if entries_cmp:
        choice_cmp = st.selectbox(
            "Task", entries_cmp, key="cmp_choice",
            format_func=lambda e: f"{e['task_id']}: {e['request'][:70]}...")
        compare_modes = st.multiselect("Policies to compare",
                                       ["reference", "random", "hf"],
                                       default=["reference", "random"])
        if st.button("Run comparison", key="cmp_run"):
            task_cmp = core.load_benchmark_task(choice_cmp["task_id"], TASKS_FILE)
            cols = st.columns(max(len(compare_modes), 1))
            for col, m in zip(cols, compare_modes):
                with col:
                    st.markdown(f"##### {m}")
                    if m == "hf":
                        mm = st.session_state.get("tf_model")
                        tk = st.session_state.get("tf_tokenizer")
                        if mm is None or tk is None:
                            st.warning("No HF model loaded (use the sidebar button); skipped.")
                            continue
                        policy, desc = core.build_policy(
                            "hf", model=mm, tokenizer=tk,
                            model_id=st.session_state.get("tf_model_id"),
                            adapter_id=st.session_state.get("tf_adapter_id"),
                            compact_catalog=bool(st.session_state.get("tf_compact", True)))
                    else:
                        policy, desc = core.build_policy(m, seed=int(seed), task=task_cmp)
                    st.caption(desc)
                    with st.spinner(f"Running {m}..."):
                        outcome = core.run_demo(task_cmp, policy)
                    render_outcome(outcome, key_prefix=m)

with tab_tools:
    st.subheader("Tool catalog & action grammar")
    st.write(render_grammar_help)
    tool_family = st.selectbox("Family", core.available_families(), key="tools_family")
    env = core.ENVIRONMENT_CLASSES[tool_family](seed=0)
    for spec in env.tools:
        with st.expander(f"`{spec.name}` — {spec.description[:90]}..."):
            st.json(spec.to_dict())
    st.markdown("#### Internal executor vs MCP adapter")
    st.write("This demo uses the **internal executor** (in-process routing, the same code "
             "path as benchmarking and training). The optional **MCP adapter** "
             "(`python -m toolforge.mcp_adapter <family>`) exposes the identical tools to "
             "external MCP clients; the core benchmark never depends on MCP.")
