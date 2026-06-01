# CLV & Churn Risk Engine

End-to-end pipeline for scoring e-commerce customers by **Customer Lifetime Value (CLV)** and **Churn Risk**, with a Power BI-ready output table and full experiment tracking via MLflow.

---

## What it does

| Stage | Description |
|---|---|
| **Ingest** | Loads orders from a database or generates synthetic data |
| **Validate** | Runs data quality checks with Great Expectations |
| **RFM Features** | Builds Recency, Frequency, Monetary quintile scores |
| **Churn Features** | Engineers 10+ behavioural predictors + churn label |
| **CLV Model** | BG/NBD + Gamma-Gamma probabilistic model (12-month horizon) |
| **Churn Model** | XGBoost classifier with SHAP explainability |
| **Export** | Merges scores → `.csv` / `.xlsx` ready for Power BI |

---

## Architecture

```
orders (DB or synthetic)
        |
    [ingest] --> data/raw/orders.parquet
        |
    [validate]
        |
    [rfm] ---------> data/processed/rfm.parquet
        |
    [churn_features] -> data/processed/churn_features.parquet
        |          |
   [clv_model]  [churn_model]
        |          |
        +----+-----+
             |
     data/processed/scored_customers.parquet
             |
      [export_powerbi]
             |
     data/processed/powerbi_customers.csv / .xlsx
```

---

## Quickstart

**Prerequisites:** Python 3.11+

```bash
# 1. Clone and set up environment
git clone https://github.com/Mkdittak/CLV-Churn-Risk-Engine.git
cd CLV-Churn-Risk-Engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
make run

# 4. (Optional) View experiment results
make mlflow
# then open http://localhost:5000
```

> **No database?** Leave `DB_CONNECTION = ""` in `config/settings.py` and the pipeline auto-generates 2,000 synthetic customers.

---

## Configuration

All constants live in `config/settings.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `SNAPSHOT_DATE` | `"2024-12-31"` | Reference "today" — never use `datetime.today()` |
| `CHURN_WINDOW_DAYS` | `90` | Days of inactivity = churned |
| `CLV_HORIZON_MONTHS` | `12` | Prediction horizon |
| `DB_CONNECTION` | `""` | SQLAlchemy connection string (empty = synthetic data) |

To connect a real database, set `DB_CONNECTION` to a SQLAlchemy URL:

```python
DB_CONNECTION = "postgresql://user:password@localhost:5432/mydb"
```

---

## Models

### CLV — BG/NBD + Gamma-Gamma
- Predicts expected purchases and average order value over the next 12 months
- `penalizer_coef=0.01` on both fitters
- Serialised with `dill` (joblib cannot pickle lifetimes objects)

### Churn — XGBoost Classifier
- Binary label: no purchase within `CHURN_WINDOW_DAYS` of `SNAPSHOT_DATE`
- Handles class imbalance via `scale_pos_weight`
- SHAP summary plot saved to `models/shap_summary.png`

**Target metrics:**

| Metric | Target |
|---|---|
| AUROC | >= 0.80 |
| KS Statistic | >= 0.35 |
| Precision @ top decile | >= 2x base churn rate |

---

## Power BI Output

`data/processed/powerbi_customers.csv` contains one row per customer:

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `churn_risk_score` | Model probability (0–1) |
| `predicted_12m_revenue` | Expected CLV over next 12 months |
| `rfm_segment` | RFM quintile label |
| `days_since_last_purchase` | Recency |
| `total_orders` | Frequency |
| `total_revenue` | Monetary |
| `churn_risk_band` | High / Medium / Low |
| `clv_tier` | High / Mid / Low Value |

Churn band thresholds: **High** >= 0.70, **Medium** >= 0.40, **Low** < 0.40.

---

## Project Structure

```
CLV-Churn-Risk-Engine/
├── config/
│   └── settings.py          # All constants and paths
├── src/
│   ├── data/
│   │   ├── ingest.py         # Data loading / synthetic generation
│   │   └── validate.py       # Great Expectations checks
│   ├── features/
│   │   ├── rfm.py            # RFM feature engineering
│   │   └── churn_features.py # Churn predictors + label
│   ├── models/
│   │   ├── clv_model.py      # BG/NBD + Gamma-Gamma
│   │   └── churn_model.py    # XGBoost + SHAP
│   └── utils/
│       └── helpers.py
├── app/
│   └── export_powerbi.py     # Final merge + CSV/XLSX export
├── dags/
│   └── clv_churn_dag.py      # Airflow DAG (daily @ 02:00 UTC)
├── tests/
│   └── test_features.py
├── data/
│   ├── raw/                  # orders.parquet (git-ignored)
│   └── processed/            # Intermediate + final outputs (git-ignored)
├── models/                   # Serialised model files (git-ignored)
├── Makefile
└── requirements.txt
```

---

## Make Commands

```bash
make run      # Full pipeline end-to-end
make train    # Re-train models only
make test     # Run pytest
make mlflow   # Launch MLflow UI on :5000
make clean    # Delete processed data and model artefacts
```

---

## Scheduling with Airflow

The DAG `clv_churn_pipeline` in `dags/clv_churn_dag.py` runs the full pipeline daily at 02:00 UTC. Set up Airflow and point it at this directory to enable automated scoring.

---

## License

MIT
