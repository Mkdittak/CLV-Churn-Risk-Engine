# Computes Recency, Frequency, Monetary features and quintile scores per customer

import pandas as pd

from config.settings import DATA_PROC, DATA_RAW, RFM_BINS, SNAPSHOT_DATE


def build_rfm(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features for every customer.

    Only orders on or before SNAPSHOT_DATE are used, preventing any data
    leakage from future transactions.

    Columns produced
    ----------------
    recency          : days since last order relative to SNAPSHOT_DATE
    frequency        : number of distinct orders
    monetary         : total revenue across all orders
    T                : customer age — days from first order to SNAPSHOT_DATE
                       (required by the lifetimes BG/NBD model)
    avg_monetary     : monetary / frequency (used by Gamma-Gamma fitter)
    R_score          : recency quintile 1–5 (5 = most recent)
    F_score          : frequency quintile 1–5 (5 = most orders)
    M_score          : monetary quintile 1–5 (5 = highest spend)
    rfm_segment      : three-digit string concatenation of R, F, M scores

    Parameters
    ----------
    orders_df : pd.DataFrame
        Raw orders with at minimum: customer_id, order_id, order_date, revenue.

    Returns
    -------
    pd.DataFrame
        One row per customer with all RFM columns.
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    # Only use orders up to the snapshot date — no future data
    df = orders_df[orders_df["order_date"] <= snapshot].copy()

    rfm = (
        df.groupby("customer_id")
        .agg(
            last_order_date=("order_date", "max"),
            first_order_date=("order_date", "min"),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (snapshot - rfm["last_order_date"]).dt.days
    rfm["T"] = (snapshot - rfm["first_order_date"]).dt.days
    rfm["avg_monetary"] = rfm["monetary"] / rfm["frequency"]

    # Quintile scoring — use rank to handle ties and edge cases
    # Recency: lower days = more recent = better = higher score
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), RFM_BINS, labels=False)
    rfm["R_score"] = (RFM_BINS - 1 - rfm["R_score"]).astype(int) + 1  # invert: low recency → high score

    rfm["F_score"] = (
        pd.qcut(rfm["frequency"].rank(method="first"), RFM_BINS, labels=False).astype(int) + 1
    )
    rfm["M_score"] = (
        pd.qcut(rfm["monetary"].rank(method="first"), RFM_BINS, labels=False).astype(int) + 1
    )

    rfm["rfm_segment"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )

    # Drop intermediate date columns before saving
    rfm = rfm.drop(columns=["last_order_date", "first_order_date"])

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(DATA_PROC / "rfm.parquet", index=False)
    return rfm


if __name__ == "__main__":
    orders = pd.read_parquet(DATA_RAW / "orders.parquet")
    rfm = build_rfm(orders)
    print(f"Customers: {len(rfm):,}")
    print("\nTop 5 segments by customer count:")
    print(rfm["rfm_segment"].value_counts().head())
