import json
import sqlite3
from typing import Union

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from pypdf import PdfReader


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
