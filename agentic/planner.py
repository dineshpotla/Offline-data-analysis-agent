import json
from typing import Any, Dict, Optional

from agentic.config import get_llm


def _parse_json_maybe(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def plan_query(user_query: str, schema: str, memory_context: Optional[str] = None) -> Dict[str, Any]:
    llm = get_llm()
    memory_block = f"\nPrevious context:\n{memory_context}\n" if memory_context else ""
    prompt = f"""
You are a Planner Agent for offline EDA.

User request:
{user_query}

Schema:
{schema}
{memory_block}

Produce a JSON plan with steps. Each step has:
- action (one of: load_file, summarize, describe_columns, find_nulls, compute_statistics, filter_rows, run_sql, correlation, summarize_text, categorical_distributions, outlier_report, plot_correlation, plot_distributions, plot_outliers, plotly_correlation, plotly_distributions, plotly_outliers, run_python)
- args (where needed)
Use run_python when the user asks to generate or save files beyond the built-in actions.

Return JSON only.
"""
    raw = llm(prompt)
    parsed = _parse_json_maybe(raw)
    if parsed:
        return parsed

    retry_prompt = f"""
Return ONLY a valid JSON object. Do not include any other text.

User request:
{user_query}

Schema:
{schema}

Actions: load_file, summarize, describe_columns, find_nulls, compute_statistics,
filter_rows, run_sql, correlation, summarize_text, categorical_distributions,
outlier_report, plot_correlation, plot_distributions, plot_outliers,
plotly_correlation, plotly_distributions, plotly_outliers, run_python.
"""
    raw_retry = llm(retry_prompt)
    parsed_retry = _parse_json_maybe(raw_retry)
    if parsed_retry:
        return parsed_retry
    raise ValueError("Planner failed to return valid JSON.")


def plan_auto_eda(file_path: str) -> Dict[str, Any]:
    return {
        "steps": [
            {"action": "load_file", "args": {"path": file_path}},
            {"action": "describe_columns"},
            {"action": "find_nulls"},
            {"action": "compute_statistics"},
            {"action": "categorical_distributions"},
            {"action": "correlation"},
            {"action": "outlier_report"},
        ]
    }
