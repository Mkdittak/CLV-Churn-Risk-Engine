# Runs data quality checks on the raw orders DataFrame

import pandas as pd

from config.settings import DATA_RAW, SNAPSHOT_DATE


def run_validation(df: pd.DataFrame) -> None:
    """
    Run data quality checks on the orders DataFrame.

    Checks performed (in order):
    1. No null values in customer_id, order_id, order_date, revenue.
    2. Revenue is always positive (> 0).
    3. No order dates after SNAPSHOT_DATE.
    4. No empty-string customer_id or order_id.
    5. At least 1,000 distinct customers.

    Prints PASS or FAIL for each check. Raises ValueError if any check fails.

    Parameters
    ----------
    df : pd.DataFrame
        Raw orders DataFrame loaded from orders.parquet.

    Raises
    ------
    ValueError
        If one or more checks fail. Message lists all failing check names.
    """
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    results: dict[str, bool] = {}

    # 1. No nulls in key columns
    key_cols = ["customer_id", "order_id", "order_date", "revenue"]
    results["no_nulls_in_key_columns"] = df[key_cols].isnull().sum().sum() == 0

    # 2. Revenue always positive
    results["revenue_always_positive"] = bool((df["revenue"] > 0).all())

    # 3. No order dates after snapshot
    results["no_orders_after_snapshot_date"] = bool((df["order_date"] <= snapshot).all())

    # 4. No empty-string IDs
    no_empty = all(
        (df[c].astype(str).str.strip() != "").all()
        for c in ["customer_id", "order_id"]
    )
    results["no_empty_string_ids"] = no_empty

    # 5. At least 1,000 distinct customers
    results["at_least_1000_distinct_customers"] = df["customer_id"].nunique() >= 1_000

    failed = []
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check}")
        if not passed:
            failed.append(check)

    if failed:
        raise ValueError(f"Data validation failed — checks not passed: {failed}")

    print("All validation checks passed.")


if __name__ == "__main__":
    df = pd.read_parquet(DATA_RAW / "orders.parquet")
    run_validation(df)
