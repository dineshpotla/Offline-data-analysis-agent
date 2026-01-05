from typing import Any, Dict, List


def generate_code(plan: Dict[str, Any]) -> str:
    """
    Produce Python code that executes the plan using pandas/duckdb.
    This is primarily for transparency; executor can also run plan directly.
    """
    lines: List[str] = [
        "import duckdb",
        "import pandas as pd",
        "from agentic.tools.loaders import load_file",
        "from agentic.tools import eda",
        "",
        "result = None",
    ]

    for step in plan.get("steps", []):
        action = step.get("action")
        args = step.get("args", {}) or {}
        if action == "load_file":
            lines.append(f'df = load_file("{args.get("path", "")}")')
        elif action == "summarize":
            lines.append("result = df.head()")
        elif action == "describe_columns":
            lines.append("result = df.dtypes")
        elif action == "find_nulls":
            lines.append("result = df.isnull().sum()")
        elif action == "compute_statistics":
            lines.append("result = df.describe(include='all')")
        elif action == "filter_rows":
            cond = args.get("condition", "")
            lines.append(f'result = df.query("{cond}")')
        elif action == "run_sql":
            q = args.get("query", "")
            lines.append("con = duckdb.connect()")
            lines.append("con.register('df', df)")
            lines.append(f'result = con.execute("""{q}""").fetchdf()')
            lines.append("con.unregister('df')")
            lines.append("con.close()")
        elif action == "correlation":
            lines.append("result = df.select_dtypes(include='number').corr()")
        elif action == "summarize_text":
            lines.append("result = df[:1200] if isinstance(df, str) else str(df)[:1200]")
        elif action == "categorical_distributions":
            lines.append("result = eda.categorical_distributions(df)")
        elif action == "outlier_report":
            lines.append("result = eda.detect_outliers_iqr(df)")
        else:
            lines.append(f"# Unsupported action: {action}")

    lines.append("output = result")
    return "\n".join(lines)
