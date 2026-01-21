"""
Local Streamlit UI for the agentic offline EDA platform.

Run:
  streamlit run ui_streamlit.py
"""

import tempfile
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st

from agentic.orchestrator import AgenticOrchestrator


st.set_page_config(page_title="Offline Agentic EDA", layout="wide")
st.title("Offline Agentic EDA")

st.sidebar.header("Chats")

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None


def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "messages": [],
        "memory_path": f".agentic_session_{chat_id}.jsonl",
    }
    st.session_state.active_chat_id = chat_id


if st.sidebar.button("New chat"):
    new_chat()

if st.session_state.chats:
    chat_options = list(st.session_state.chats.keys())
    labels = [st.session_state.chats[c]["title"] for c in chat_options]
    selected = st.sidebar.radio("Conversations", chat_options, format_func=lambda x: st.session_state.chats[x]["title"])
    st.session_state.active_chat_id = selected
else:
    st.sidebar.info("Create a chat to begin.")

st.sidebar.header("Settings")
auto_eda = st.sidebar.checkbox("Auto-EDA on first run", value=True)
max_rounds = st.sidebar.number_input("Max reflection rounds", min_value=1, max_value=10, value=3)
st.sidebar.caption("Model: LFM2-2.6B (set LFM2_PATH if not in ./models/LFM2-2.6B)")

uploaded = st.file_uploader("Upload dataset (CSV, Excel, JSON, Parquet, SQLite, PDF)", type=None)

if "file_path" not in st.session_state:
    st.session_state.file_path = None


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
if clear_btn and st.session_state.active_chat_id:
    st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
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


active_chat_id = st.session_state.active_chat_id
messages = st.session_state.chats.get(active_chat_id, {}).get("messages", [])
for msg in messages:
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
        if not active_chat_id:
            new_chat()
            active_chat_id = st.session_state.active_chat_id
        chat = st.session_state.chats[active_chat_id]
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        orchestrator = AgenticOrchestrator(session_memory_path=chat["memory_path"])
        res = orchestrator.run(
            prompt,
            st.session_state.file_path,
            rounds=max_rounds,
            auto_eda=auto_eda,
        )

        assistant_text = res.report or "Done."
        chat["messages"].append({"role": "assistant", "content": assistant_text})
        chat["messages"].append({"role": "assistant", "type": "result", "content": res.result})

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
