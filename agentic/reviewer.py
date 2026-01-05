from typing import Any, Tuple

import pandas as pd


def review_result(result: Any) -> Tuple[bool, str]:
    """Return (ok, feedback)."""
    if result is None:
        return False, "Result is None."
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return False, "DataFrame is empty."
        if any(result.dtypes.isnull()):
            return False, "DataFrame has invalid dtypes."
    if hasattr(result, "empty") and getattr(result, "empty"):
        return False, "Result object is empty."
    return True, "Result looks valid."
