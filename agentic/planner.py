import json
import re
from typing import Any, Dict, Optional

from agentic.config import get_llm


def _intent_plan(user_query: str) -> Optional[Dict[str, Any]]:
    q = user_query.lower().strip()
    if re.search(r"\b(how many|count|number of)\b.*\b(rows|records)\b", q):
        return {
            "steps": [
                {"action": "load_file", "args": {"path": ""}},
                {"action": "run_sql", "args": {"query": "SELECT COUNT(*) AS row_count FROM df"}},
            ]
        }
    if re.search(r"\b(attribute|column|field)\b.*\b(names|list)\b", q):
        return {
            "steps": [
                {"action": "load_file", "args": {"path": ""}},
                {"action": "describe_columns"},
            ]
        }
    return None


def plan_query(user_query: str, schema: str, memory_context: Optional[str] = None) -> Dict[str, Any]:
    llm = get_llm()
    memory_block = f"\nPrevious context:\n{memory_context}\n" if memory_context else ""
    intent_plan = _intent_plan(user_query)
    if intent_plan:
        return intent_plan
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
    return json.loads(raw)


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
