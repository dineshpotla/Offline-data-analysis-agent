import pandas as pd


def profile_basic(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum().sort_values(ascending=False)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all")


def categorical_distributions(df: pd.DataFrame, top_n: int = 10) -> dict:
    dist = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        dist[col] = df[col].value_counts().head(top_n)
    return dist


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr()


def detect_outliers_iqr(df: pd.DataFrame, k: float = 1.5) -> dict:
    """Simple IQR-based outlier detection per numeric column."""
    out = {}
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
        out[col] = int(mask.sum())
    return out
