"""
config.py — Project-wide constants and configuration.
"""

import os

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ── Validation ─────────────────────────────────────────────────
VALIDATION_DAYS = 30          # last N days of training data for validation
TOP_N_COMBINATIONS = 10       # store-family combos for Prophet / TimesFM

# ── Random Forest hyper-parameters ─────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_TRAIN_SAMPLES = 500_000
RF_RANDOM_STATE = 42

# ── TimesFM ────────────────────────────────────────────────────
TIMESFM_HORIZON = 15
TIMESFM_BATCH_SIZE = 32
TIMESFM_CHECKPOINT = "google/timesfm-1.0-200m-pytorch"

# ── Feature columns used by the ML models ──────────────────────
FEATURE_COLS = [
    "store_nbr", "onpromotion", "cluster",
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end", "is_payday", "days_to_payday",
    "earthquake", "days_since_earthquake",
    "dcoilwtico", "oil_ma7", "oil_ma30",
    "is_national_holiday", "is_regional_holiday", "is_transferred_holiday",
    "transactions",
    "sales_lag7", "sales_lag14", "sales_lag30",
    "sales_rolling_mean7", "sales_rolling_mean14", "sales_rolling_std7",
    "family_encoded", "city_encoded", "state_encoded", "type_encoded",
]
