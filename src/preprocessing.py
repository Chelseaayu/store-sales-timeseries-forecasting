"""
preprocessing.py — Feature engineering, encoding, and train/val split.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import FEATURE_COLS, VALIDATION_DAYS


# ── Date features ──────────────────────────────────────────────
def _create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and payday features derived from the ``date`` column."""
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_payday"] = ((df["day"] == 15) | (df["is_month_end"] == 1)).astype(int)
    df["days_to_payday"] = df["day"].apply(lambda x: min(abs(x - 15), abs(x - 30)))
    return df


# ── Main pipeline ──────────────────────────────────────────────
def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build all features on *train* and *test* DataFrames in-place and return
    (train, test, label_encoders).
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Date features
    train = _create_date_features(train)
    test = _create_date_features(test)
    print("[INFO] Date features created!")

    # Earthquake indicator
    eq_date = pd.to_datetime("2016-04-16")
    for df in (train, test):
        df["earthquake"] = (df["date"] >= eq_date).astype(int)
        df["days_since_earthquake"] = (df["date"] - eq_date).dt.days.clip(lower=0)
    print("[INFO] Earthquake features added!")

    # Store metadata
    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")
    print("[INFO] Store features merged!")

    # Oil prices
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil["oil_ma7"] = oil["dcoilwtico"].rolling(window=7, min_periods=1).mean()
    oil["oil_ma30"] = oil["dcoilwtico"].rolling(window=30, min_periods=1).mean()

    train = train.merge(oil[["date", "dcoilwtico", "oil_ma7", "oil_ma30"]], on="date", how="left")
    test = test.merge(oil[["date", "dcoilwtico", "oil_ma7", "oil_ma30"]], on="date", how="left")

    for df in (train, test):
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
        df["oil_ma7"] = df["oil_ma7"].ffill().bfill()
        df["oil_ma30"] = df["oil_ma30"].ffill().bfill()
    print("[INFO] Oil price features merged!")

    # Holiday flags
    national = holidays[holidays["locale"] == "National"]["date"].unique()
    regional = holidays[holidays["locale"] == "Regional"]["date"].unique()
    transferred = holidays[holidays["transferred"] == True]["date"].unique()

    for df in (train, test):
        df["is_national_holiday"] = df["date"].isin(national).astype(int)
        df["is_regional_holiday"] = df["date"].isin(regional).astype(int)
        df["is_transferred_holiday"] = df["date"].isin(transferred).astype(int)
    print("[INFO] Holiday features added!")

    # Transactions
    train = train.merge(transactions, on=["date", "store_nbr"], how="left")
    test = test.merge(transactions, on=["date", "store_nbr"], how="left")

    median_tx = train.groupby("store_nbr")["transactions"].transform("median")
    train["transactions"] = train["transactions"].fillna(median_tx)
    test["transactions"] = test["transactions"].fillna(
        train.groupby("store_nbr")["transactions"].transform("median")
    )
    print("[INFO] Transaction features merged!")

    # Lag / rolling features (aggregated daily)
    daily_agg = (
        train.sort_values("date")
        .groupby("date")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "total_daily_sales"})
    )

    daily_agg["sales_lag7"] = daily_agg["total_daily_sales"].shift(7)
    daily_agg["sales_lag14"] = daily_agg["total_daily_sales"].shift(14)
    daily_agg["sales_lag30"] = daily_agg["total_daily_sales"].shift(30)
    daily_agg["sales_rolling_mean7"] = daily_agg["total_daily_sales"].rolling(7, min_periods=1).mean()
    daily_agg["sales_rolling_mean14"] = daily_agg["total_daily_sales"].rolling(14, min_periods=1).mean()
    daily_agg["sales_rolling_std7"] = daily_agg["total_daily_sales"].rolling(7, min_periods=1).std()

    lag_cols = [
        "date", "sales_lag7", "sales_lag14", "sales_lag30",
        "sales_rolling_mean7", "sales_rolling_mean14", "sales_rolling_std7",
    ]
    train = train.merge(daily_agg[lag_cols], on="date", how="left")
    test = test.merge(daily_agg[lag_cols], on="date", how="left")

    for col in lag_cols[1:]:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)
    print("[INFO] Lag and rolling features created!")

    # Categorical encoding
    cat_cols = ["family", "city", "state", "type"]
    label_encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col + "_encoded"] = le.fit_transform(train[col])
        test[col + "_encoded"] = le.transform(test[col])
        label_encoders[col] = le
    print("[INFO] Categorical encoding completed!")

    print(f"\nFinal train shape: {train.shape}")
    print(f"Final test shape : {test.shape}")

    return train, test, label_encoders


# ── Prepare X / y / split ──────────────────────────────────────
def prepare_data(train: pd.DataFrame):
    """
    Create feature matrix, target vector, and time-based train/val split.

    Returns
    -------
    X, y, X_train, y_train, X_val, y_val, split_date
    """
    X = train[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
    y = train["sales"]

    split_date = train["date"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_mask = train["date"] <= split_date
    val_mask = train["date"] > split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\n[INFO] Feature matrix shape : {X.shape}")
    print(f"Training set  : {X_train.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Split date    : {split_date}")

    return X, y, X_train, y_train, X_val, y_val, split_date
