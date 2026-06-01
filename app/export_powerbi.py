# Formats the scored customer table for Power BI ingestion and saves CSV and XLSX outputs

import numpy as np
import pandas as pd

from config.settings import DATA_PROC


def export_for_powerbi() -> pd.DataFrame:
    """
    Load scored_customers.parquet, reshape for business readability, and
    export as both CSV and XLSX for Power BI ingestion.

    Column renames
    --------------
    customer_id   → Customer ID
    churn_prob    → Churn Risk Score
    clv_12m       → Predicted 12M Revenue
    rfm_segment   → RFM Segment
    recency       → Days Since Last Purchase
    frequency     → Total Orders
    monetary      → Total Revenue

    Derived columns
    ---------------
    Churn Risk Band : High (≥ 0.70) / Medium (≥ 0.40) / Low (< 0.40)
    CLV Tier        : High Value / Mid Value / Low Value (equal-size tertiles)

    Returns
    -------
    pd.DataFrame
        Business-ready DataFrame that was written to disk.
    """
    scored = pd.read_parquet(DATA_PROC / "scored_customers.parquet")

    col_map = {
        "customer_id": "Customer ID",
        "churn_prob": "Churn Risk Score",
        "clv_12m": "Predicted 12M Revenue",
        "rfm_segment": "RFM Segment",
        "recency": "Days Since Last Purchase",
        "frequency": "Total Orders",
        "monetary": "Total Revenue",
    }
    out = scored[list(col_map.keys())].rename(columns=col_map).copy()

    # Churn Risk Band
    conditions = [
        out["Churn Risk Score"] >= 0.70,
        out["Churn Risk Score"] >= 0.40,
    ]
    out["Churn Risk Band"] = np.select(conditions, ["High", "Medium"], default="Low")

    # CLV Tier — equal-size tertiles
    out["CLV Tier"] = pd.qcut(
        out["Predicted 12M Revenue"].rank(method="first"),
        q=3,
        labels=["Low Value", "Mid Value", "High Value"],
    )

    # Round all float columns
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(2)

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_PROC / "powerbi_customers.csv", index=False)
    out.to_excel(DATA_PROC / "powerbi_customers.xlsx", index=False, engine="openpyxl")

    # Summary
    total = len(out)
    print(f"\nTotal customers: {total:,}")
    print("\nCustomers by Churn Risk Band:")
    print(out["Churn Risk Band"].value_counts().to_string())
    print("\nPredicted 12M Revenue by Churn Risk Band:")
    print(
        out.groupby("Churn Risk Band")["Predicted 12M Revenue"]
        .sum()
        .map(lambda x: f"£{x:,.2f}")
        .to_string()
    )

    return out


if __name__ == "__main__":
    export_for_powerbi()
