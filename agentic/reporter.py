import json
from typing import Any

import pandas as pd

from agentic.config import get_llm


def summarize_result(result: Any, query: str, schema: str) -> str:
    llm = get_llm()
    if isinstance(result, pd.DataFrame):
        preview = result.head().to_markdown()
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
