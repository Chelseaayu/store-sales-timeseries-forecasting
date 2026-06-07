"""
main.py — Main orchestrator script. Tying loading, preprocessing, modeling, evaluation, and prediction.
"""

import os
import argparse
import pandas as pd
import numpy as np

from config import DATA_DIR
from data_loader import load_data
from eda import run_eda
from preprocessing import engineer_features, prepare_data
from models import train_baseline, train_random_forest, train_prophet, train_timesfm
from evaluation import compare_models, plot_feature_importance, plot_actual_vs_predicted, print_insights
from predict import generate_submission


def main():
    parser = argparse.ArgumentParser(description="Store Sales Time Series Forecasting Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"Path to the folder containing raw CSV data files. Defaults to workspace folder: {DATA_DIR}",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip running full Exploratory Data Analysis (EDA) plotting to save time.",
    )
    parser.add_argument(
        "--skip-prophet",
        action="store_true",
        help="Force skip training Prophet models (even if the package is installed).",
    )
    parser.add_argument(
        "--skip-timesfm",
        action="store_true",
        help="Force skip running TimesFM model (even if the package is installed).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" FAVORITA STORE SALES FORECASTING - MACHINE LEARNING PIPELINE")
    print("=" * 65)
    print("Portfolio Project by: Chelsea Ayu")
    print("LinkedIn: https://linkedin.com/in/chelseaayu")
    print("Portfolio: https://chelsea-ayu.vercel.app/")
    print("=" * 65)

    # 1. Resolve path and load raw datasets
    data_path = args.data_path or DATA_DIR
    data = load_data(data_path)
    
    # Keep fresh copies of train/test since EDA might add temporary columns
    train_raw = data["train"].copy()
    test_raw = data["test"].copy()

    # 2. Run Exploratory Data Analysis (EDA) or compute quick correlation stats
    if not args.skip_eda:
        daily_sales, oil_corr, tx_corr, holiday_sales = run_eda(
            train_raw, data["stores"], data["oil"], data["holidays"], data["transactions"]
        )
    else:
        print("\n[INFO] Skipping EDA plotting...")
        # Compute baseline correlation values for final insights print
        daily_sales = train_raw.groupby("date")["sales"].sum().reset_index()
        oil_df = data["oil"].copy()
        oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
        daily_sales_oil = daily_sales.merge(oil_df, on="date", how="left")
        oil_corr = float(daily_sales_oil["sales"].corr(daily_sales_oil["dcoilwtico"]))
        
        sales_by_sd = train_raw.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
        tx_sales = data["transactions"].merge(sales_by_sd, on=["date", "store_nbr"], how="inner")
        tx_corr = float(tx_sales["transactions"].corr(tx_sales["sales"]))
        
        train_holiday = train_raw.copy()
        train_holiday["is_holiday"] = train_holiday["date"].isin(data["holidays"]["date"]).astype(int)
        holiday_sales = train_holiday.groupby("is_holiday")["sales"].mean()

    # 3. Perform Feature Engineering
    # We pass fresh copies of the original datasets
    train_feats, test_feats, label_encoders = engineer_features(
        data["train"].copy(),
        data["test"].copy(),
        data["stores"].copy(),
        data["oil"].copy(),
        data["holidays"].copy(),
        data["transactions"].copy(),
    )

    # 4. Data Preparation (Feature matrix and Train/Val split)
    X, y, X_train, y_train, X_val, y_val, split_date = prepare_data(train_feats)

    # 5. Model Training & Validation
    # A. Baseline
    baseline_pred, _, _ = train_baseline(y_train, y_val)

    # B. Random Forest
    rf_model, rf_pred_val, _, _ = train_random_forest(X_train, y_train, X_val, y_val)
    feature_importance = plot_feature_importance(rf_model, X_train.columns.tolist())
    plot_actual_vs_predicted(y_val, rf_pred_val)

    # Identify top combinations for time series models
    train_agg = train_feats.groupby(["store_nbr", "family"])["sales"].sum()
    top_combinations = train_agg.nlargest(10).index.tolist()

    # C. Prophet
    prophet_val_df = pd.DataFrame()
    if not args.skip_prophet:
        _, prophet_val_df, _, _ = train_prophet(train_feats, split_date, top_combinations)
    else:
        print("\n[INFO] Skipping Prophet models training...")

    # D. TimesFM
    timesfm_val_df = pd.DataFrame()
    if not args.skip_timesfm:
        timesfm_val_df, _, _ = train_timesfm(train_feats, split_date, top_combinations)
    else:
        print("\n[INFO] Skipping TimesFM forecasting...")

    # 6. Evaluation & Comparison
    comparison = compare_models(
        train=train_feats,
        X=X,
        y=y,
        y_train=y_train,
        split_date=split_date,
        top_combinations=top_combinations,
        rf_model=rf_model,
        prophet_val_df=prophet_val_df,
        timesfm_val_df=timesfm_val_df,
    )

    # 7. Generate submission file on test set using the main model
    generate_submission(rf_model, test_feats, data_path)

    # 8. Print business insights report
    print_insights(
        comparison=comparison,
        feature_importance=feature_importance,
        oil_correlation=oil_corr,
        transaction_correlation=tx_corr,
        holiday_sales=holiday_sales,
        train=train_feats,
    )


if __name__ == "__main__":
    main()
