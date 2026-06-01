# Builds additional churn predictor features and the binary churn label

import pandas as pd

from config.settings import CHURN_WINDOW_DAYS, DATA_PROC, DATA_RAW, SNAPSHOT_DATE


def _inter_purchase_time(dates: pd.Series) -> float:
    """
    Compute mean days between consecutive orders for a single customer.

    Returns CHURN_WINDOW_DAYS for customers with only one order, since
    there is no inter-purchase interval to observe.
    """
    sorted_dates = dates.sort_values()
    if len(sorted_dates) < 2:
        return float(CHURN_WINDOW_DAYS)
    diffs = sorted_dates.diff().dropna().dt.days
    return float(diffs.mean())


def build_churn_features(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full churn feature table, including the binary churn label.

    Loads the RFM table from DATA_PROC/rfm.parquet and merges it with
    orders_df to compute additional per-customer behavioural features.

    Feature definitions
    -------------------
    days_since_first_order  : customer tenure in days (same as T from RFM)
    avg_order_value         : mean revenue per order
    std_order_value         : std of revenue per order (0 for single-order customers)
    inter_purchase_time_avg : mean days between consecutive orders;
                              defaults to CHURN_WINDOW_DAYS for single-order customers
    promo_dependency_ratio  : placeholder 0.0 — replace with real promo data when available
    return_rate             : fraction of orders that were returned

    Churn label
    -----------
    churn_label = 1 if recency > CHURN_WINDOW_DAYS, else 0.

    Since we are working with historical data and SNAPSHOT_DATE is a fixed
    point in the past, "recency" (days since last order relative to SNAPSHOT_DATE)
    is fully observable. A customer whose last purchase was more than
    CHURN_WINDOW_DAYS before SNAPSHOT_DATE is treated as churned — they had
    ample opportunity to return during that window and did not.
    This avoids any dependency on future data and prevents data leakage.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Raw orders DataFrame (same as passed to build_rfm).

    Returns
    -------
    pd.DataFrame
        One row per customer with all churn features and churn_label.
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    # Use only pre-snapshot orders for feature computation
    df = orders_df[orders_df["order_date"] <= snapshot].copy()

    rfm = pd.read_parquet(DATA_PROC / "rfm.parquet")

    # --- Additional per-customer aggregations ---
    agg = (
        df.groupby("customer_id")
        .agg(
            avg_order_value=("revenue", "mean"),
            std_order_value=("revenue", "std"),
            return_rate=("is_returned", "mean"),
        )
        .reset_index()
    )
    agg["std_order_value"] = agg["std_order_value"].fillna(0.0)

    ipt = (
        df.groupby("customer_id")["order_date"]
        .apply(_inter_purchase_time)
        .reset_index()
        .rename(columns={"order_date": "inter_purchase_time_avg"})
    )

    features = rfm.merge(agg, on="customer_id", how="left")
    features = features.merge(ipt, on="customer_id", how="left")

    features["days_since_first_order"] = features["T"]

    # Promo dependency ratio: not available in base data.
    # Replace with real data when available (e.g. promo_orders / total_orders).
    features["promo_dependency_ratio"] = 0.0

    # Churn label — fully observable from historical data, no future information needed
    features["churn_label"] = (features["recency"] > CHURN_WINDOW_DAYS).astype(int)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    features.to_parquet(DATA_PROC / "churn_features.parquet", index=False)

    n_churned = features["churn_label"].sum()
    n_total = len(features)
    print(f"Churned:  {n_churned:,} ({100 * n_churned / n_total:.1f}%)")
    print(f"Retained: {n_total - n_churned:,} ({100 * (n_total - n_churned) / n_total:.1f}%)")

    return features


if __name__ == "__main__":
    orders = pd.read_parquet(DATA_RAW / "orders.parquet")
    build_churn_features(orders)
