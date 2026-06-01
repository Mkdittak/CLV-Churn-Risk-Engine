# Trains an XGBoost churn classifier and computes SHAP feature importances

import matplotlib
matplotlib.use("Agg")  # headless rendering — must be set before importing pyplot
import matplotlib.pyplot as plt

import joblib
import mlflow
import numpy as np

from config.settings import DATA_PROC, MODELS_DIR, RANDOM_STATE, TEST_SIZE

mlflow.set_tracking_uri("sqlite:///mlflow.db")

import pandas as pd
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "R_score",
    "F_score",
    "M_score",
    "days_since_first_order",
    "avg_order_value",
    "std_order_value",
    "inter_purchase_time_avg",
    "return_rate",
]


def _precision_at_top_decile(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return precision within the top 10% of predicted churn probabilities."""
    n_top = max(1, int(len(y_prob) * 0.10))
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    return float(y_true[top_idx].mean())


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the KS (Kolmogorov-Smirnov) statistic — maximum separation
    between the cumulative distribution functions of churned and non-churned
    predicted probabilities.
    """
    churned_probs = np.sort(y_prob[y_true == 1])
    retained_probs = np.sort(y_prob[y_true == 0])
    all_thresholds = np.unique(y_prob)
    ks = 0.0
    for t in all_thresholds:
        tpr = (churned_probs >= t).mean()
        fpr = (retained_probs >= t).mean()
        ks = max(ks, abs(tpr - fpr))
    return ks


def train_churn_model(features_df: pd.DataFrame):
    """
    Train an XGBoost churn classifier and produce a fully scored customer table.

    Steps
    -----
    1. Random stratified split and time-based split (most recent 20% as test).
    2. Fit StandardScaler + XGBClassifier with early stopping.
    3. Evaluate on both test sets; log AUROC, KS, precision@decile to MLflow.
    4. Compute SHAP values and save summary plot.
    5. Save pipeline to MODELS_DIR/churn_pipeline.joblib.
    6. Add churn_prob to features_df, merge with clv_scores.parquet,
       save scored_customers.parquet to DATA_PROC.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of build_churn_features — one row per customer.

    Returns
    -------
    tuple[Pipeline, np.ndarray, pd.DataFrame]
        Fitted sklearn Pipeline, SHAP values array, X_test DataFrame.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = features_df.dropna(subset=FEATURES + ["churn_label"]).copy()
    X = df[FEATURES]
    y = df["churn_label"].values

    # --- Random stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Time-based split: most recent 20% of customers as test set ---
    # Customers with lowest recency (fewest days since last order) are most recent.
    n_time_test = max(1, int(len(df) * TEST_SIZE))
    time_test_idx = df.nsmallest(n_time_test, "recency").index
    X_test_time = X.loc[time_test_idx]
    y_test_time = y[df.index.get_indexer(time_test_idx)]
    X_train_time = X.drop(time_test_idx)
    y_train_time = np.delete(y, df.index.get_indexer(time_test_idx))

    # --- Scale features (manual, for early stopping eval_set compatibility) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_test_time_s = scaler.transform(X_test_time)

    # Class imbalance weight
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    xgb_clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
    )
    xgb_clf.fit(
        X_train_s,
        y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    # --- Evaluation ---
    y_prob = xgb_clf.predict_proba(X_test_s)[:, 1]
    y_prob_time = xgb_clf.predict_proba(X_test_time_s)[:, 1]

    auroc_random = roc_auc_score(y_test, y_prob)
    auroc_time = roc_auc_score(y_test_time, y_prob_time)
    ks = _ks_statistic(y_test, y_prob)
    prec_decile = _precision_at_top_decile(y_test, y_prob)

    print(classification_report(y_test, xgb_clf.predict(X_test_s)))
    print(f"AUROC (random split):    {auroc_random:.4f}")
    print(f"AUROC (time-based split): {auroc_time:.4f}")
    print(f"KS statistic:            {ks:.4f}")
    print(f"Precision @ top decile:  {prec_decile:.4f}")

    with mlflow.start_run(run_name="churn_model"):
        mlflow.log_params({
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(scale_pos_weight, 4),
        })
        mlflow.log_metrics({
            "auroc_random_split": auroc_random,
            "auroc_time_split": auroc_time,
            "ks_statistic": ks,
            "precision_at_top_decile": prec_decile,
        })

    # --- SHAP ---
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_s)

    shap.summary_plot(shap_values, X_test_s, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved to {MODELS_DIR / 'shap_summary.png'}")

    # --- Save pipeline ---
    pipeline = Pipeline([("scaler", scaler), ("xgbclassifier", xgb_clf)])
    joblib.dump(pipeline, MODELS_DIR / "churn_pipeline.joblib")

    # --- Score all customers and merge with CLV ---
    all_scaled = scaler.transform(X)
    df = df.copy()
    df["churn_prob"] = xgb_clf.predict_proba(all_scaled)[:, 1]

    clv = pd.read_parquet(DATA_PROC / "clv_scores.parquet")
    scored = df.merge(
        clv[["customer_id", "predicted_purchases", "expected_avg_revenue", "clv_12m"]],
        on="customer_id",
        how="left",
    )

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(DATA_PROC / "scored_customers.parquet", index=False)
    print(f"Scored customers saved: {len(scored):,} rows")

    return pipeline, shap_values, X_test


if __name__ == "__main__":
    features = pd.read_parquet(DATA_PROC / "churn_features.parquet")
    train_churn_model(features)
