import json
from typing import Any, Dict

from agentic.config import get_llm


def plan_query(user_query: str, schema: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = f"""
You are a Planner Agent for offline EDA.

User request:
{user_query}

Schema:
{schema}

Produce a JSON plan with steps. Each step has:
- action (one of: load_file, summarize, describe_columns, find_nulls, compute_statistics, filter_rows, run_sql, correlation, summarize_text, categorical_distributions, outlier_report)
- args (where needed)

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
