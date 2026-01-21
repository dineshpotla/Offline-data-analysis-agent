"""
Local Streamlit UI for the agentic offline EDA platform.

Run:
  streamlit run ui_streamlit.py
"""

import tempfile
from pathlib import Path

import pandas as pd
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
if "messages" not in st.session_state:
    st.session_state.messages = []


def persist_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


if uploaded:
    st.session_state.file_path = persist_upload(uploaded)
    st.success(f"Loaded file: {uploaded.name}")

st.subheader("Chat")
show_details = st.sidebar.checkbox("Show plan/details", value=False)
clear_btn = st.sidebar.button("Clear chat")
if clear_btn:
    st.session_state.messages = []
    st.experimental_rerun()


def render_result(result):
    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
        return
    if isinstance(result, pd.Series):
        st.dataframe(result.to_frame("value"))
        return
    if isinstance(result, dict):
        st.json(result)
        return
    st.code(str(result)[:4000])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "result":
            render_result(msg["content"])
        else:
            st.markdown(msg["content"])

prompt = st.chat_input("Ask anything about your data…")
if prompt:
    if not st.session_state.file_path:
        st.error("Please upload a dataset first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        res = st.session_state.orchestrator.run(
            prompt,
            st.session_state.file_path,
            rounds=max_rounds,
            auto_eda=auto_eda and use_chat,
        )

        assistant_text = res.report or "Done."
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.messages.append({"role": "assistant", "type": "result", "content": res.result})

        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            with st.expander("Show raw result", expanded=False):
                render_result(res.result)
            if show_details:
                st.markdown("**Plan**")
                st.json(res.plan)
                st.markdown(f"**Status:** ok={res.ok}, rounds={res.rounds_used}")
                st.markdown("**Feedback**")
                st.write(res.feedback)
