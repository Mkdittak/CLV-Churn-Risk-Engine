"""
Airflow DAG: clv_churn_pipeline
================================
Orchestrates the full CLV and Churn Risk scoring pipeline on a daily cadence.

Schedule : 02:00 UTC every day
Catchup  : disabled (only runs for today's date, not missed historic runs)

Task dependency chain
---------------------
ingest → validate → rfm → churn_features → [clv_model, churn_model] → export

The CLV model and churn model run in parallel after churn_features completes,
since neither depends on the other's output until the final export step.

Manual trigger
--------------
    airflow dags trigger clv_churn_pipeline

Viewing logs
------------
Open http://localhost:8080, navigate to DAGs → clv_churn_pipeline,
click any run, then click a task → Logs.
"""

from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from config.settings import DATA_PROC, DATA_RAW

# ---------------------------------------------------------------------------
# Task callables — each wrapper loads its own input data from disk so that
# Airflow can serialise the task without passing DataFrames through XCom.
# ---------------------------------------------------------------------------

def _ingest():
    from src.data.ingest import generate_synthetic_data, load_orders
    from config.settings import DB_CONNECTION
    if DB_CONNECTION:
        load_orders(DB_CONNECTION)
    else:
        generate_synthetic_data()


def _validate():
    from src.data.validate import run_validation
    df = pd.read_parquet(DATA_RAW / "orders.parquet")
    run_validation(df)


def _rfm():
    from src.features.rfm import build_rfm
    df = pd.read_parquet(DATA_RAW / "orders.parquet")
    build_rfm(df)


def _churn_features():
    from src.features.churn_features import build_churn_features
    df = pd.read_parquet(DATA_RAW / "orders.parquet")
    build_churn_features(df)


def _clv_model():
    from src.models.clv_model import fit_clv
    rfm = pd.read_parquet(DATA_PROC / "rfm.parquet")
    fit_clv(rfm)


def _churn_model():
    from src.models.churn_model import train_churn_model
    features = pd.read_parquet(DATA_PROC / "churn_features.parquet")
    train_churn_model(features)


def _export():
    from app.export_powerbi import export_for_powerbi
    export_for_powerbi()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "data-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="clv_churn_pipeline",
    description="Daily CLV and churn risk scoring pipeline",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["clv", "churn", "ml"],
) as dag:

    task_ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    task_validate = PythonOperator(task_id="validate", python_callable=_validate)
    task_rfm = PythonOperator(task_id="rfm", python_callable=_rfm)
    task_churn_features = PythonOperator(task_id="churn_features", python_callable=_churn_features)
    task_clv_model = PythonOperator(task_id="clv_model", python_callable=_clv_model)
    task_churn_model = PythonOperator(task_id="churn_model", python_callable=_churn_model)
    task_export = PythonOperator(task_id="export", python_callable=_export)

    (
        task_ingest
        >> task_validate
        >> task_rfm
        >> task_churn_features
        >> [task_clv_model, task_churn_model]
        >> task_export
    )
