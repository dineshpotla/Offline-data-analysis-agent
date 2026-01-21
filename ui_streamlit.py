"""
Local Streamlit UI for the agentic offline EDA platform.

Run:
  streamlit run ui_streamlit.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from agentic.orchestrator import AgenticOrchestrator


st.set_page_config(page_title="Offline Agentic EDA", layout="wide")
st.title("Offline Agentic EDA")

st.sidebar.header("Settings")
auto_eda = st.sidebar.checkbox("Auto-EDA on first run", value=True)
max_rounds = st.sidebar.number_input("Max reflection rounds", min_value=1, max_value=10, value=3)
use_chat = st.sidebar.checkbox("Chat mode (remember context)", value=True)
st.sidebar.caption("Model: LFM2-2.6B (set LFM2_PATH if not in ./models/LFM2-2.6B)")

uploaded = st.file_uploader("Upload dataset (CSV, Excel, JSON, Parquet, SQLite, PDF)", type=None)

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgenticOrchestrator()
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "history" not in st.session_state:
    st.session_state.history = []


def persist_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


if uploaded:
    st.session_state.file_path = persist_upload(uploaded)
    st.success(f"Loaded file: {uploaded.name}")

prompt = st.text_area("Ask a question", placeholder="e.g., summarize dataset, plot correlation, export csv")

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Run")
clear_btn = col2.button("Clear history")

if clear_btn:
    st.session_state.history = []
    st.experimental_rerun()

if run_btn:
    if not st.session_state.file_path:
        st.error("Please upload a dataset first.")
    elif not prompt.strip():
        st.error("Please enter a question.")
    else:
        res = st.session_state.orchestrator.run(
            prompt.strip(),
            st.session_state.file_path,
            rounds=max_rounds,
            auto_eda=auto_eda,
        )
        st.session_state.history.append({"prompt": prompt, "result": res})

st.subheader("Results")
for item in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {item['prompt']}")
    res = item["result"]
    st.markdown(f"**Status:** ok={res.ok}, rounds={res.rounds_used}")
    st.markdown("**Plan:**")
    st.json(res.plan)
    st.markdown("**Feedback:**")
    st.write(res.feedback)
    st.markdown("**Result preview:**")
    st.code(str(res.result)[:2000])
    if res.report:
        st.markdown("**Report:**")
        st.write(res.report)
