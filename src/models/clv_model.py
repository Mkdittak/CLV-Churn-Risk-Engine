# Fits BG/NBD and Gamma-Gamma models to predict 12-month Customer Lifetime Value

import dill
import joblib
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

import mlflow

from config.settings import CLV_HORIZON_MONTHS, DATA_PROC, DATA_RAW, MODELS_DIR

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def fit_clv(rfm_df: pd.DataFrame):
    """
    Fit a probabilistic CLV model (BG/NBD + Gamma-Gamma) and score all customers.

    Steps
    -----
    1. Fit BetaGeoFitter on the full RFM table.
    2. Fit GammaGammaFitter on repeat-purchaser subset (frequency > 1).
    3. Predict expected purchases over CLV_HORIZON_MONTHS * 30 days.
    4. Predict expected average order value.
    5. Compute clv_12m = predicted_purchases * expected_avg_revenue.
    6. Save enriched DataFrame to DATA_PROC/clv_scores.parquet.
    7. Persist both model objects to MODELS_DIR with joblib.
    8. Log parameters and metrics to MLflow.

    BG/NBD column conventions (lifetimes library)
    ----------------------------------------------
    - frequency  : number of REPEAT purchases (total orders - 1)
    - recency    : days from first to last purchase (T - days_since_last_order)
    - T          : customer age in days from first order to SNAPSHOT_DATE

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Output of build_rfm — one row per customer.

    Returns
    -------
    tuple[pd.DataFrame, BetaGeoFitter, GammaGammaFitter]
        Enriched DataFrame with CLV columns, fitted BGF, fitted GGF.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    penalizer = 0.01
    horizon_days = CLV_HORIZON_MONTHS * 30

    df = rfm_df.copy()

    # Convert to lifetimes BG/NBD conventions
    # - bgf_frequency: repeat purchases (total - 1, floored at 0)
    # - bgf_recency:   age at last purchase = T - days_since_last_order
    df["bgf_frequency"] = (df["frequency"] - 1).clip(lower=0)
    df["bgf_recency"] = df["T"] - df["recency"]

    with mlflow.start_run(run_name="clv_model"):
        mlflow.log_param("penalizer_coef", penalizer)
        mlflow.log_param("horizon_months", CLV_HORIZON_MONTHS)

        # --- BG/NBD model ---
        bgf = BetaGeoFitter(penalizer_coef=penalizer)
        bgf.fit(df["bgf_frequency"], df["bgf_recency"], df["T"])

        df["predicted_purchases"] = bgf.predict(
            horizon_days,
            df["bgf_frequency"],
            df["bgf_recency"],
            df["T"],
        )

        # --- Gamma-Gamma model (repeat purchasers only) ---
        # GGF requires frequency > 0 (i.e., at least one repeat purchase)
        repeat_mask = df["bgf_frequency"] > 0
        ggf = GammaGammaFitter(penalizer_coef=penalizer)
        ggf.fit(df.loc[repeat_mask, "bgf_frequency"], df.loc[repeat_mask, "avg_monetary"])

        df["expected_avg_revenue"] = 0.0
        df.loc[repeat_mask, "expected_avg_revenue"] = ggf.conditional_expected_average_profit(
            df.loc[repeat_mask, "bgf_frequency"],
            df.loc[repeat_mask, "avg_monetary"],
        )
        # Single-purchase customers: use their observed avg_monetary as best estimate
        df.loc[~repeat_mask, "expected_avg_revenue"] = df.loc[~repeat_mask, "avg_monetary"]

        df["clv_12m"] = df["predicted_purchases"] * df["expected_avg_revenue"]

        # --- Log metrics ---
        mlflow.log_metric("mean_clv_12m", df["clv_12m"].mean())
        mlflow.log_metric("median_clv_12m", df["clv_12m"].median())

        # --- Persist models (dill handles lifetimes' internal lambda functions) ---
        with open(MODELS_DIR / "bgf.pkl", "wb") as f:
            dill.dump(bgf, f)
        with open(MODELS_DIR / "ggf.pkl", "wb") as f:
            dill.dump(ggf, f)
        mlflow.log_artifact(str(MODELS_DIR / "bgf.pkl"))
        mlflow.log_artifact(str(MODELS_DIR / "ggf.pkl"))

    # Drop temporary BG/NBD helper columns before saving
    df = df.drop(columns=["bgf_frequency", "bgf_recency"])

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROC / "clv_scores.parquet", index=False)

    print(f"CLV model fitted. Mean CLV 12m:   {df['clv_12m'].mean():.2f}")
    print(f"Median CLV 12m:                   {df['clv_12m'].median():.2f}")

    return df, bgf, ggf


if __name__ == "__main__":
    rfm = pd.read_parquet(DATA_PROC / "rfm.parquet")
    fit_clv(rfm)
