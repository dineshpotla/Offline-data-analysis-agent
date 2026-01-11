from typing import Any, Tuple

import numpy as np
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
        # NaN / inf checks
        nan_counts = result.isnull().sum().sum()
        if nan_counts > 0 and nan_counts > 0.2 * result.size:
            return False, f"Too many missing values ({nan_counts})."
        if np.isinf(result.select_dtypes(include="number")).to_numpy().any():
            return False, "Contains infinite values."
    if hasattr(result, "empty") and getattr(result, "empty"):
        return False, "Result object is empty."
    return True, "Result looks valid."
