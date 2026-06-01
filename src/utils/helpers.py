# Shared utility functions used across the pipeline

import pandas as pd

from config.settings import SNAPSHOT_DATE


def get_snapshot_timestamp() -> pd.Timestamp:
    """
    Return SNAPSHOT_DATE as a pandas Timestamp.

    Use this instead of re-parsing the string in every module.

    Returns
    -------
    pd.Timestamp
    """
    return pd.Timestamp(SNAPSHOT_DATE)


def assert_no_future_dates(df: pd.DataFrame, date_col: str = "order_date") -> None:
    """
    Raise ValueError if any date in date_col is after SNAPSHOT_DATE.

    Use this as a quick leakage guard at the start of feature functions.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
        Name of the datetime column to check.

    Raises
    ------
    ValueError
        If future dates are found.
    """
    snapshot = get_snapshot_timestamp()
    future = df[df[date_col] > snapshot]
    if not future.empty:
        raise ValueError(
            f"{len(future)} rows in '{date_col}' are after SNAPSHOT_DATE ({SNAPSHOT_DATE}). "
            "Filter the DataFrame before computing features."
        )


def log_dataframe_summary(df: pd.DataFrame, label: str = "") -> None:
    """
    Print a brief summary of a DataFrame: shape, nulls, dtypes.

    Parameters
    ----------
    df : pd.DataFrame
    label : str
        Optional label printed before the summary.
    """
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Shape: {df.shape}")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"{prefix}Nulls:\n{null_counts[null_counts > 0]}")
    else:
        print(f"{prefix}No nulls found.")
