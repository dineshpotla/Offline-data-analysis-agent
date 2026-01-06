import contextlib
import io
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

from agentic.tools import eda
from agentic.tools.loaders import load_file


class ExecutionContext:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.text: Optional[str] = None
        self.sqlite_conn = None


def safe_exec(code: str, ctx: ExecutionContext) -> Dict[str, Any]:
    """Run generated code with a minimal globals set."""
    glob = {
        "duckdb": duckdb,
        "pd": pd,
        "load_file": load_file,
        "eda": eda,
    }
    loc = {"df": ctx.df, "result": None}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, glob, loc)
    return {"result": loc.get("output") or loc.get("result"), "stdout": buffer.getvalue()}


def execute_plan(plan: Dict[str, Any], ctx: ExecutionContext) -> Any:
    steps = plan.get("steps", [])
    result: Any = None

    for step in steps:
        action = step.get("action")
        args = step.get("args", {}) or {}

        if action == "load_file":
            path = args.get("path")
            loaded = load_file(path)
            if isinstance(loaded, pd.DataFrame):
                ctx.df = loaded
            elif isinstance(loaded, str):
                ctx.text = loaded
            result = loaded

        elif action == "summarize":
            if ctx.df is not None:
                result = ctx.df.head()
            elif ctx.text:
                result = ctx.text[:1000]

        elif action == "describe_columns":
            result = ctx.df.dtypes if ctx.df is not None else None

        elif action == "find_nulls":
            result = ctx.df.isnull().sum() if ctx.df is not None else None

        elif action == "compute_statistics":
            result = ctx.df.describe(include="all") if ctx.df is not None else None

        elif action == "filter_rows":
            cond = args.get("condition", "")
            result = ctx.df.query(cond) if ctx.df is not None else None

        elif action == "run_sql":
            query = args.get("query", "")
            if ctx.df is not None:
                con = duckdb.connect()
                con.register("df", ctx.df)
                result = con.execute(query).fetchdf()
                con.unregister("df")
                con.close()

        elif action == "correlation":
            if ctx.df is not None:
                numeric_df = ctx.df.select_dtypes(include="number")
                result = numeric_df.corr() if not numeric_df.empty else None

        elif action == "summarize_text":
            result = ctx.text[:1200] if ctx.text else None

        elif action == "categorical_distributions":
            result = eda.categorical_distributions(ctx.df) if ctx.df is not None else None

        elif action == "outlier_report":
            result = eda.detect_outliers_iqr(ctx.df) if ctx.df is not None else None

        elif action == "plot_correlation":
            if ctx.df is not None:
                path = args.get("path", "outputs/correlation.png")
                result = eda.plot_correlation_heatmap(ctx.df, path=path)

        elif action == "plot_distributions":
            if ctx.df is not None:
                path = args.get("path", "outputs/distributions.png")
                cols = args.get("cols")
                bins = args.get("bins", 30)
                result = eda.plot_distributions(ctx.df, cols=cols, path=path, bins=bins)

        elif action == "plot_outliers":
            if ctx.df is not None:
                path = args.get("path", "outputs/outliers.png")
                cols = args.get("cols")
                result = eda.plot_outlier_boxplots(ctx.df, cols=cols, path=path)

        else:
            raise ValueError(f"Unsupported action: {action}")

    return result
