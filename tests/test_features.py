# Pytest unit tests for RFM feature engineering and churn label logic

import numpy as np
import pandas as pd
import pytest

from config.settings import CHURN_WINDOW_DAYS, SNAPSHOT_DATE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def patch_data_proc(tmp_path, monkeypatch):
    """
    Redirect DATA_PROC to a temporary directory so tests never write to
    the real data/processed folder and are fully isolated from each other.
    """
    monkeypatch.setattr("config.settings.DATA_PROC", tmp_path)
    monkeypatch.setattr("src.features.rfm.DATA_PROC", tmp_path)
    monkeypatch.setattr("src.features.churn_features.DATA_PROC", tmp_path)
    return tmp_path


@pytest.fixture()
def small_orders_df():
    """
    A minimal 10-row synthetic orders DataFrame covering 3 distinct customers
    with clearly different recency, frequency, and monetary profiles.

    Customer 1001 — most recent, high frequency, high spend
    Customer 1002 — mid recency, low frequency, low spend
    Customer 1003 — oldest (least recent), single order, low spend
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    rows = [
        # customer_id, order_id, order_date, revenue, channel, is_returned
        (1001, 1,  snapshot - pd.Timedelta(days=5),   80.0, "email",   0),
        (1001, 2,  snapshot - pd.Timedelta(days=20),  60.0, "organic", 0),
        (1001, 3,  snapshot - pd.Timedelta(days=45),  90.0, "email",   1),
        (1001, 4,  snapshot - pd.Timedelta(days=70),  50.0, "organic", 0),
        (1002, 5,  snapshot - pd.Timedelta(days=50),  30.0, "social",  0),
        (1002, 6,  snapshot - pd.Timedelta(days=100), 25.0, "social",  0),
        (1003, 7,  snapshot - pd.Timedelta(days=200), 15.0, "email",   0),
        (1003, 8,  snapshot - pd.Timedelta(days=250), 12.0, "organic", 1),
        (1003, 9,  snapshot - pd.Timedelta(days=300), 10.0, "email",   0),
        (1003, 10, snapshot - pd.Timedelta(days=350), 11.0, "paid_search", 0),
    ]
    return pd.DataFrame(rows, columns=[
        "customer_id", "order_id", "order_date", "revenue", "channel", "is_returned"
    ])


@pytest.fixture()
def churn_label_orders_df():
    """
    Two-customer DataFrame designed to test churn label logic precisely.

    - Customer 9001: last order 200 days before SNAPSHOT_DATE → should be
      labelled churned (200 > CHURN_WINDOW_DAYS).
    - Customer 9002: last order 10 days before SNAPSHOT_DATE → should be
      labelled retained (10 < CHURN_WINDOW_DAYS).
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    rows = [
        (9001, 101, snapshot - pd.Timedelta(days=200), 40.0, "email",   0),
        (9001, 102, snapshot - pd.Timedelta(days=400), 35.0, "organic", 0),
        (9002, 103, snapshot - pd.Timedelta(days=10),  55.0, "social",  0),
        (9002, 104, snapshot - pd.Timedelta(days=90),  45.0, "email",   0),
    ]
    return pd.DataFrame(rows, columns=[
        "customer_id", "order_id", "order_date", "revenue", "channel", "is_returned"
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rfm_recency_scores_correct(patch_data_proc, small_orders_df):
    """
    The customer with the most recent purchase should receive a higher
    R_score than the customer with the oldest purchase.

    Customer 1001 last ordered 5 days ago; customer 1003 last ordered
    200 days ago. After quintile scoring (higher = more recent), 1001
    must have a strictly higher R_score than 1003.
    """
    from src.features.rfm import build_rfm

    rfm = build_rfm(small_orders_df).set_index("customer_id")

    assert rfm.loc[1001, "R_score"] > rfm.loc[1003, "R_score"], (
        "Customer 1001 (5 days ago) should have a higher R_score than "
        "customer 1003 (200 days ago)."
    )


def test_rfm_no_nulls(patch_data_proc, small_orders_df):
    """
    After running build_rfm, the scored columns R_score, F_score, M_score,
    and rfm_segment must contain zero null values for every customer.

    Null scores indicate a broken quintile calculation or a merge issue.
    """
    from src.features.rfm import build_rfm

    rfm = build_rfm(small_orders_df)
    for col in ["R_score", "F_score", "M_score", "rfm_segment"]:
        null_count = rfm[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null values."


def test_rfm_monetary_is_sum(patch_data_proc, small_orders_df):
    """
    The monetary value for a known customer must equal the sum of their
    individual order revenues from the raw orders DataFrame.

    This confirms that monetary aggregation is a simple sum, not a mean
    or any other aggregation.
    """
    from src.features.rfm import build_rfm

    rfm = build_rfm(small_orders_df).set_index("customer_id")
    expected_monetary_1001 = small_orders_df[
        small_orders_df["customer_id"] == 1001
    ]["revenue"].sum()

    assert np.isclose(rfm.loc[1001, "monetary"], expected_monetary_1001), (
        f"Customer 1001 monetary {rfm.loc[1001, 'monetary']:.2f} "
        f"!= expected sum {expected_monetary_1001:.2f}"
    )


def test_churn_label_logic(patch_data_proc, churn_label_orders_df):
    """
    Churn label logic: a customer inactive for > CHURN_WINDOW_DAYS gets
    label=1; a customer who purchased within CHURN_WINDOW_DAYS gets label=0.

    Customer 9001 (last order 200 days ago) → churned (1).
    Customer 9002 (last order 10 days ago)  → retained (0).
    """
    from src.features.rfm import build_rfm
    from src.features.churn_features import build_churn_features

    # build_churn_features depends on rfm.parquet being present
    build_rfm(churn_label_orders_df)
    features = build_churn_features(churn_label_orders_df).set_index("customer_id")

    assert features.loc[9001, "churn_label"] == 1, (
        "Customer 9001 (inactive 200 days) should be labelled churned."
    )
    assert features.loc[9002, "churn_label"] == 0, (
        "Customer 9002 (purchased 10 days ago) should be labelled retained."
    )


def test_feature_no_data_leakage(patch_data_proc, small_orders_df):
    """
    No feature in the RFM output should be influenced by orders placed
    after SNAPSHOT_DATE.

    We append a future order for customer 1001 and verify that their
    recency, frequency, and monetary values are identical to the output
    without the future order, proving that build_rfm filters to
    SNAPSHOT_DATE internally.
    """
    from src.features.rfm import build_rfm

    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    # Baseline: RFM without any future orders
    rfm_baseline = build_rfm(small_orders_df).set_index("customer_id")

    # Add a future order for customer 1001
    future_order = pd.DataFrame([{
        "customer_id": 1001,
        "order_id": 999,
        "order_date": snapshot + pd.Timedelta(days=30),  # AFTER snapshot
        "revenue": 9999.0,
        "channel": "email",
        "is_returned": 0,
    }])
    orders_with_future = pd.concat([small_orders_df, future_order], ignore_index=True)

    rfm_with_future = build_rfm(orders_with_future).set_index("customer_id")

    # Recency, frequency, monetary must be unchanged despite the future order
    for col in ["recency", "frequency", "monetary"]:
        assert np.isclose(
            rfm_baseline.loc[1001, col], rfm_with_future.loc[1001, col]
        ), (
            f"Data leakage detected: '{col}' changed when a post-snapshot "
            f"order was present. build_rfm must filter to <= SNAPSHOT_DATE."
        )
