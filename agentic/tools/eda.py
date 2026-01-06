import math
import os

import matplotlib

# Force non-GUI backend for headless/offline use
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


def _ensure_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame, path: str = "outputs/correlation.png") -> str | None:
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    if corr.empty:
        return None

    _ensure_dir(path)
    size = max(6, corr.shape[0] * 0.8)
    plt.figure(figsize=(size, size))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    path: str = "outputs/distributions.png",
    bins: int = 30,
) -> str | None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return None
    if cols:
        numeric_df = numeric_df[[c for c in cols if c in numeric_df.columns]]
    if numeric_df.empty:
        return None

    n = len(numeric_df.columns)
    cols_per_row = 3
    rows = math.ceil(n / cols_per_row)
    plt.figure(figsize=(4 * cols_per_row, 3 * rows))
    for idx, col in enumerate(numeric_df.columns):
        ax = plt.subplot(rows, cols_per_row, idx + 1)
        numeric_df[col].dropna().hist(bins=bins, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    _ensure_dir(path)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_outlier_boxplots(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    path: str = "outputs/outliers.png",
) -> str | None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return None
    if cols:
        numeric_df = numeric_df[[c for c in cols if c in numeric_df.columns]]
    if numeric_df.empty:
        return None

    _ensure_dir(path)
    plt.figure(figsize=(max(6, len(numeric_df.columns) * 1.5), 4))
    numeric_df.boxplot(rot=45)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
