"""
Offline data-analysis agent with local planning (Phi-4 placeholder),
execution over tabular/text data, and simple verification.

Supported inputs: CSV, Excel, JSON, Parquet, SQLite DB, PDF.
No internet or external frameworks required.

Quick start:
  pip install pandas numpy pyarrow openpyxl duckdb tabulate pypdf
  python offline_agent.py "top 5 customers by revenue" sales.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from pypdf import PdfReader
from tabulate import tabulate

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# ---------------------------------------------------------------------------
# Model hooks (replace with your offline models)
# ---------------------------------------------------------------------------


def phi4_llm(prompt: str) -> str:
    """
    Call local Phi-4 Mini Q6 via llama-cpp-python.

    Set env PHI4_MINI_Q6_PATH to the GGUF file path.
    Example: export PHI4_MINI_Q6_PATH=~/models/phi-4-mini-q6.gguf
    """
    if Llama is None:
        raise ImportError(
            "llama-cpp-python not installed. Install with `pip install llama-cpp-python`."
        )

    model_path = os.getenv("PHI4_MINI_Q6_PATH")
    if not model_path:
        raise EnvironmentError("Set PHI4_MINI_Q6_PATH to your phi-4-mini-q6 GGUF file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Lazy init to avoid reload per call
    if not hasattr(phi4_llm, "_llm"):
        phi4_llm._llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )

    llm: Llama = phi4_llm._llm
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response["choices"][0]["message"]["content"]


def roberta_embed(texts: List[str]) -> List[List[float]]:
    """
    Optional: plug your offline RoBERTa embedding model here.
    Provided as a stub so you can wire retrieval if desired.
    """
    raise NotImplementedError("Plug in your offline RoBERTa embedding model here.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_file(path: str) -> Union[pd.DataFrame, str, sqlite3.Connection]:
    """Load supported file types into memory."""
    if path.endswith(".csv"):
        return pd.read_csv(path)

    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)

    if path.endswith(".json"):
        return pd.read_json(path)

    if path.endswith(".parquet"):
        table = pq.read_table(path)
        return table.to_pandas()

    if path.endswith(".db") or path.endswith(".sqlite"):
        return sqlite3.connect(path)

    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text_chunks = []
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks)

    raise ValueError(f"Unsupported file format: {path}")


def infer_schema(data: Union[pd.DataFrame, str, sqlite3.Connection]) -> str:
    """Return a lightweight schema string for planner conditioning."""
    if isinstance(data, pd.DataFrame):
        return str(data.dtypes)

    if isinstance(data, sqlite3.Connection):
        cursor = data.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        pieces = []
        for tbl in tables:
            cursor.execute(f"PRAGMA table_info('{tbl}')")
            cols = [f"{r[1]}:{r[2]}" for r in cursor.fetchall()]
            pieces.append(f"{tbl} -> {', '.join(cols)}")
        return "; ".join(pieces) if pieces else "empty_sqlite_db"

    if isinstance(data, str):
        preview = data[:400].replace("\n", " ")
        return f"pdf_text_preview: {preview}"

    return "unknown_schema"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def plan_with_phi4(user_query: str, schema: str) -> Dict[str, Any]:
    """Ask Phi-4 to produce a JSON plan."""
    prompt = f"""
You are an offline data-analysis agent.

User request:
{user_query}

Dataset schema:
{schema}

Output a JSON plan with fields:
steps: list of ordered actions.
Each action is one of:
- load_file
- summarize
targets: df for tabular, text for pdf
- describe_columns
- find_nulls
- filter_rows
- run_sql
- compute_statistics
- correlation
- summarize_text
- extract_tables_text

Include arguments where needed. Return JSON only.
"""
    raw = phi4_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.error("Planner returned invalid JSON: %s", raw)
        raise ValueError("Planner JSON parse failed") from exc


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------


@dataclass
class ExecutionContext:
    df: Optional[pd.DataFrame] = None
    text: Optional[str] = None
    sqlite_conn: Optional[sqlite3.Connection] = None


def execute_plan(plan: Dict[str, Any], context: ExecutionContext) -> Any:
    """Execute a planner-produced plan against provided context."""
    steps = plan.get("steps", [])
    result: Any = None

    for step in steps:
        action = step.get("action")
        args = step.get("args", {}) or {}

        if action == "load_file":
            path = args.get("path")
            if not path:
                raise ValueError("load_file requires 'path'")
            loaded = load_file(path)
            if isinstance(loaded, pd.DataFrame):
                context.df = loaded
                context.text = None
                context.sqlite_conn = None
            elif isinstance(loaded, sqlite3.Connection):
                context.sqlite_conn = loaded
                context.df = None
                context.text = None
            elif isinstance(loaded, str):
                context.text = loaded
                context.df = None
                context.sqlite_conn = None
            else:
                raise ValueError(f"Unsupported loaded type: {type(loaded)}")
            result = loaded

        elif action == "summarize":
            if context.df is not None:
                result = context.df.head()
            elif context.text:
                result = context.text[:1000]
            else:
                raise ValueError("summarize needs df or text")

        elif action == "describe_columns":
            if context.df is None:
                raise ValueError("describe_columns requires DataFrame")
            result = context.df.dtypes

        elif action == "find_nulls":
            if context.df is None:
                raise ValueError("find_nulls requires DataFrame")
            result = context.df.isnull().sum()

        elif action == "compute_statistics":
            if context.df is None:
                raise ValueError("compute_statistics requires DataFrame")
            result = context.df.describe(include="all")

        elif action == "filter_rows":
            if context.df is None:
                raise ValueError("filter_rows requires DataFrame")
            condition = args.get("condition")
            if not condition:
                raise ValueError("filter_rows needs 'condition'")
            result = context.df.query(condition)

        elif action == "run_sql":
            query = args.get("query")
            if not query:
                raise ValueError("run_sql needs 'query'")
            if context.df is not None:
                con = duckdb.connect()
                con.register("df", context.df)
                result = con.execute(query).fetchdf()
                con.unregister("df")
                con.close()
            elif context.sqlite_conn is not None:
                result = pd.read_sql_query(query, context.sqlite_conn)
            else:
                raise ValueError("run_sql requires DataFrame or SQLite connection")

        elif action == "correlation":
            if context.df is None:
                raise ValueError("correlation requires DataFrame")
            numeric_df = context.df.select_dtypes(include="number")
            if numeric_df.empty:
                raise ValueError("No numeric columns for correlation")
            result = numeric_df.corr()

        elif action == "summarize_text":
            if not context.text:
                raise ValueError("summarize_text requires text content")
            # Placeholder: could call a local summarizer here.
            result = context.text[:1200]

        else:
            raise ValueError(f"Unsupported action: {action}")

    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_result(res: Any) -> bool:
    """Lightweight sanity checks on execution output."""
    if res is None:
        return False
    if isinstance(res, pd.DataFrame) and res.empty:
        return False
    if hasattr(res, "empty") and getattr(res, "empty"):
        return False
    return True


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def data_agent(query: str, file_path: str) -> Any:
    """Plan → execute → verify loop."""
    data = load_file(file_path)
    schema = infer_schema(data)
    plan = plan_with_phi4(query, schema)
    ctx = ExecutionContext()
    # Seed context with the preloaded object to avoid re-reading if plan skips load.
    if isinstance(data, pd.DataFrame):
        ctx.df = data
    elif isinstance(data, sqlite3.Connection):
        ctx.sqlite_conn = data
    elif isinstance(data, str):
        ctx.text = data

    result = execute_plan(plan, ctx)
    if not verify_result(result):
        return "Plan executed but result empty or invalid."
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline data-analysis agent")
    parser.add_argument("query", help="Natural language request")
    parser.add_argument("file_path", help="Path to data file (csv/xlsx/json/parquet/sqlite/pdf)")
    args = parser.parse_args()

    query = args.query
    file_path = os.path.expanduser(args.file_path)

    result = data_agent(query, file_path)

    if isinstance(result, pd.DataFrame):
        print(tabulate(result, headers="keys", tablefmt="psql", showindex=False))
    else:
        print(result)


if __name__ == "__main__":
    main()

