import json
from typing import Any

import pandas as pd

from agentic.config import get_llm


def summarize_result(result: Any, query: str, schema: str) -> str:
    llm = get_llm()
    # Deterministic summaries for common intents
    if isinstance(result, pd.DataFrame):
        if "row_count" in result.columns and len(result) == 1:
            count = int(result["row_count"].iloc[0])
            return f"Row count: {count}"
        preview = result.head().to_markdown()
    elif isinstance(result, pd.Series):
        preview = result.to_frame("value").head().to_markdown()
    elif isinstance(result, dict):
        # Likely missing-value counts
        keys = list(result.keys())
        if keys and all(isinstance(v, (int, float)) for v in result.values()):
            return (
                "Missing values per column:\n"
                + "\n".join([f"- {k}: {v}" for k, v in result.items()])
            )
        preview = json.dumps(result, indent=2)
    else:
        preview = str(result)[:1200]
    prompt = f"""
You are a Report Agent. Summarize the analysis result for the user.

User query: {query}
Schema: {schema}

Result preview:
{preview}

Produce concise bullet insights and suggested next questions.
"""
    return llm(prompt)


def render_report(result: Any, query: str, schema: str) -> str:
    try:
        return summarize_result(result, query, schema)
    except Exception:
        # Fallback: structured dump
        try:
            return json.dumps({"result": str(result)[:1200]}, indent=2)
        except Exception:
            return "Report unavailable."
