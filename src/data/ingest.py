# Pulls raw order data from a SQL database or generates a synthetic dataset

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config.settings import DATA_RAW, DB_CONNECTION, SNAPSHOT_DATE


def load_orders(conn_str: str) -> pd.DataFrame:
    """
    Load order history from a SQL database.

    Connects via SQLAlchemy and selects customer_id, order_id, order_date,
    revenue, channel, and is_returned for all orders on or before
    SNAPSHOT_DATE. Saves the result to DATA_RAW/orders.parquet.

    Parameters
    ----------
    conn_str : str
        SQLAlchemy-compatible connection string.

    Returns
    -------
    pd.DataFrame
        Raw orders DataFrame.
    """
    engine = create_engine(conn_str)
    query = f"""
        SELECT customer_id, order_id, order_date, revenue, channel, is_returned
        FROM orders
        WHERE order_date <= '{SNAPSHOT_DATE}'
    """
    df = pd.read_sql(query, engine, parse_dates=["order_date"])
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_RAW / "orders.parquet", index=False)
    return df


def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate a realistic synthetic orders dataset for development and testing.

    Produces 50,000 orders across 8,000 customers spanning 3 years up to
    SNAPSHOT_DATE. Revenue follows a log-normal distribution (mean ~45,
    std ~30). Saves to DATA_RAW/orders.parquet and returns the DataFrame.

    Returns
    -------
    pd.DataFrame
        Synthetic orders DataFrame with columns:
        customer_id, order_id, order_date, revenue, channel, is_returned.
    """
    rng = np.random.default_rng(42)
    n_orders = 50_000
    n_customers = 8_000

    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    start_date = snapshot - pd.DateOffset(years=3)
    total_days = (snapshot - start_date).days

    customer_ids = rng.integers(1, n_customers + 1, size=n_orders)
    order_ids = np.arange(1, n_orders + 1)

    days_offset = rng.integers(0, total_days + 1, size=n_orders)
    order_dates = pd.to_datetime(
        [start_date + pd.Timedelta(days=int(d)) for d in days_offset]
    )

    # Log-normal parameters targeting mean ~45, std ~30
    mean_rev, std_rev = 45.0, 30.0
    sigma = np.sqrt(np.log(1 + (std_rev / mean_rev) ** 2))
    mu = np.log(mean_rev) - 0.5 * sigma ** 2
    revenue = rng.lognormal(mean=mu, sigma=sigma, size=n_orders).round(2)

    channels = rng.choice(["email", "organic", "paid_search", "social"], size=n_orders)
    is_returned = (rng.random(size=n_orders) < 0.05).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "order_id": order_ids,
        "order_date": order_dates,
        "revenue": revenue,
        "channel": channels,
        "is_returned": is_returned,
    })

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_RAW / "orders.parquet", index=False)
    return df


if __name__ == "__main__":
    if DB_CONNECTION:
        df = load_orders(DB_CONNECTION)
    else:
        df = generate_synthetic_data()
    print(f"Rows loaded: {len(df):,}")
    print(f"Date range:  {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print(f"Customers:   {df['customer_id'].nunique():,}")
