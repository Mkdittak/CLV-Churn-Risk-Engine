"""
Central configuration for the CLV & Churn Risk pipeline.

All pipeline constants and path definitions live here. Import from this
module rather than hardcoding values anywhere else.

IMPORTANT: SNAPSHOT_DATE must never be set to datetime.today() or
datetime.now(). Doing so causes data leakage — feature calculations
would shift with every run, making experiments non-reproducible and
contaminating churn labels with future information.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

# Treat this date as "today" for all feature calculations.
# Set to the last date in your dataset when using historical data.
SNAPSHOT_DATE = "2024-12-31"

# A customer is labelled churned if they made no purchase in this many days
# before SNAPSHOT_DATE. Adjust to your business's natural purchase cycle.
CHURN_WINDOW_DAYS = 90

# Number of quintile bins for R, F, M scoring.
RFM_BINS = 5

# Horizon for CLV prediction (months).
CLV_HORIZON_MONTHS = 12

# Train/test split fraction and random seed.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

# Leave empty to use synthetic data generation instead.
# PostgreSQL: "postgresql://user:password@localhost:5432/mydb"
# SQLite:     "sqlite:///data/raw/sample.db"
DB_CONNECTION = ""
