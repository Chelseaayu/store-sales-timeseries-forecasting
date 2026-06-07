"""
evaluation.py — Model evaluation, metrics calculation, plotting comparison, and business insights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import save_plot, rmsle

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_feature_importance(rf_model, feature_names: list[str]) -> pd.DataFrame:
    """
    Plot and print Random Forest feature importance.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores.
    """
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features["importance"], color="steelblue", edgecolor="black", linewidth=0.5)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances", fontsize=16, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_plot("10_feature_importance.png")

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    return feature_importance


def plot_actual_vs_predicted(y_val: pd.Series, rf_pred_val: np.ndarray) -> None:
    """Scatter plot of actual vs predicted values on the full validation set."""
    sample_n = min(5000, len(y_val))
    idx = np.random.choice(len(y_val), size=sample_n, replace=False)

    plt.figure(figsize=(12, 6))
    plt.scatter(y_val.iloc[idx], rf_pred_val[idx], alpha=0.3, color="teal")
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
             "r--", linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales (Random Forest)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("11_actual_vs_predicted_rf.png")


def compare_models(
    train: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series,
    split_date: pd.Timestamp,
    top_combinations: list[tuple[int, str]],
    rf_model,
    prophet_val_df: pd.DataFrame = pd.DataFrame(),
    timesfm_val_df: pd.DataFrame = pd.DataFrame(),
) -> pd.DataFrame:
    """
    Compare all available models on the same validation subset of top store-family combinations.
    
    Returns
    -------
    pd.DataFrame
        Comparison metrics.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # 1. Filter training / validation set to only the top store-family combinations
    rf_subset_mask = pd.Series(False, index=train.index)
    for store_nbr, family in top_combinations:
        combo_mask = (
            (train["store_nbr"] == store_nbr) &
            (train["family"] == family) &
            (train["date"] > split_date)
        )
        rf_subset_mask = rf_subset_mask | combo_mask

    X_val_subset = X[rf_subset_mask]
    y_val_subset = y[rf_subset_mask]

    if len(y_val_subset) == 0:
        print("[WARN] No validation samples found for top combinations. Cannot generate comparative metrics.")
        return pd.DataFrame()

    # 2. Get predictions for Baseline (Mean) on subset
    baseline_subset_pred = np.full(len(y_val_subset), y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_val_subset, baseline_subset_pred))
    baseline_mae = mean_absolute_error(y_val_subset, baseline_subset_pred)
    baseline_rmsle = rmsle(y_val_subset, baseline_subset_pred)

    # 3. Get predictions for Random Forest on subset
    rf_pred_subset = np.maximum(rf_model.predict(X_val_subset), 0)
    rf_rmse = np.sqrt(mean_squared_error(y_val_subset, rf_pred_subset))
    rf_mae = mean_absolute_error(y_val_subset, rf_pred_subset)
    rf_rmsle = rmsle(y_val_subset, rf_pred_subset)

    # 4. Build comparison table
    models_compared = ["Baseline (Mean)", "Random Forest"]
    rmses = [baseline_rmse, rf_rmse]
    maes = [baseline_mae, rf_mae]
    rmsles = [baseline_rmsle, rf_rmsle]

    # Check if Prophet was trained
    if not prophet_val_df.empty:
        prophet_rmse = np.sqrt(mean_squared_error(prophet_val_df["actual"], prophet_val_df["yhat"]))
        prophet_mae = mean_absolute_error(prophet_val_df["actual"], prophet_val_df["yhat"])
        prophet_rmsle = rmsle(prophet_val_df["actual"], prophet_val_df["yhat"])
        
        models_compared.append("Prophet")
        rmses.append(prophet_rmse)
        maes.append(prophet_mae)
        rmsles.append(prophet_rmsle)

    # Check if TimesFM was forecasted
    if not timesfm_val_df.empty:
        timesfm_rmse = np.sqrt(mean_squared_error(timesfm_val_df["actual"], timesfm_val_df["yhat"]))
        timesfm_mae = mean_absolute_error(timesfm_val_df["actual"], timesfm_val_df["yhat"])
        timesfm_rmsle = rmsle(timesfm_val_df["actual"], timesfm_val_df["yhat"])
        
        models_compared.append("TimesFM")
        rmses.append(timesfm_rmse)
        maes.append(timesfm_mae)
        rmsles.append(timesfm_rmsle)

    comparison = pd.DataFrame({
        "Model": models_compared,
        "RMSE": rmses,
        "MAE": maes,
        "RMSLE": rmsles,
    })
    
    comparison = comparison.sort_values("RMSLE").reset_index(drop=True)
    comparison.index += 1
    comparison.index.name = "Rank"

    print(f"\n(Evaluated on validation set: top {len(top_combinations)} store-family combinations)")
    print(f"(Validation period: {split_date.strftime('%Y-%m-%d')} onwards)\n")
    print(comparison.to_string())
    print(f"\nBest Model: {comparison.iloc[0]['Model']} (RMSLE: {comparison.iloc[0]['RMSLE']:.4f})")
    print("Note: RMSLE is the official Kaggle competition metric")

    # 5. Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["RMSE", "MAE", "RMSLE"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    for idx, metric in enumerate(metrics):
        bars = axes[idx].bar(
            comparison["Model"], comparison[metric],
            color=colors[: len(comparison)], edgecolor="black", linewidth=0.5,
        )
        axes[idx].set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
        axes[idx].set_ylabel(metric)
        axes[idx].tick_params(axis="x", rotation=30)
        axes[idx].grid(True, alpha=0.3, axis="y")
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.2f}", ha="center", va="bottom", fontsize=9,
            )

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot("12_model_comparison.png")

    # 6. Plot actual vs predictions overlay for a sample combination
    sample_store, sample_family = top_combinations[0]

    sample_actual = train[
        (train["store_nbr"] == sample_store) &
        (train["family"] == sample_family) &
        (train["date"] > split_date)
    ].sort_values("date")

    plt.figure(figsize=(16, 6))
    plt.plot(sample_actual["date"], sample_actual["sales"], "k-",
             linewidth=2, label="Actual", marker="o", markersize=4)

    # RF on this sample
    sample_rf_mask = (
        (train["store_nbr"] == sample_store) &
        (train["family"] == sample_family) &
        (train["date"] > split_date)
    )
    sample_rf_X = X[sample_rf_mask]
    if len(sample_rf_X) > 0:
        sample_rf_pred = np.maximum(rf_model.predict(sample_rf_X), 0)
        plt.plot(sample_actual["date"].values[: len(sample_rf_pred)], sample_rf_pred,
                 "--", linewidth=2, label="Random Forest", alpha=0.8)

    # Prophet on this sample
    if not prophet_val_df.empty:
        sample_prophet = prophet_val_df[
            (prophet_val_df["store_nbr"] == sample_store) &
            (prophet_val_df["family"] == sample_family)
        ]
        if not sample_prophet.empty:
            plt.plot(pd.to_datetime(sample_prophet["ds"]), sample_prophet["yhat"],
                     "--", linewidth=2, label="Prophet", alpha=0.8)

    # TimesFM on this sample
    if not timesfm_val_df.empty:
        sample_timesfm = timesfm_val_df[
            (timesfm_val_df["store_nbr"] == sample_store) &
            (timesfm_val_df["family"] == sample_family)
        ]
        if not sample_timesfm.empty:
            plt.plot(pd.to_datetime(sample_timesfm["ds"]), sample_timesfm["yhat"],
                     "--", linewidth=2, label="TimesFM", alpha=0.8)

    plt.title(
        f"Model Predictions Comparison\nStore {sample_store} - {sample_family}",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("13_prediction_overlay.png")

    return comparison


def print_insights(
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    oil_correlation: float,
    transaction_correlation: float,
    holiday_sales: pd.Series,
    train: pd.DataFrame,
) -> None:
    """Print business insights and project recommendations based on evaluation."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS & INSIGHTS")
    print("=" * 80)

    if comparison is not None and not comparison.empty:
        print("\nMODEL PERFORMANCE COMPARISON:")
        print(comparison.to_string())
        print(f"\nBest Model: {comparison.iloc[0]['Model']} (RMSLE: {comparison.iloc[0]['RMSLE']:.4f})")
    
    if feature_importance is not None and not feature_importance.empty:
        print("\nTOP PREDICTIVE FEATURES (Random Forest):")
        for _, row in feature_importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    print("\nBUSINESS INSIGHTS:")
    print(f"  - Oil price correlation with sales: {oil_correlation:.3f}")
    print(f"  - Transaction-sales correlation: {transaction_correlation:.3f}")
    
    if len(holiday_sales) >= 2:
        diff_pct = ((holiday_sales[1] - holiday_sales[0]) / holiday_sales[0] * 100)
        print(f"  - Holiday effect on sales: {diff_pct:.2f}% increase on holidays")
    
    print(f"  - Total product families analyzed: {train['family'].nunique()}")
    print(f"  - Total stores analyzed: {train['store_nbr'].nunique()}")

    print("\nSTRATEGIC RECOMMENDATIONS:")
    print("  1. Focus inventory planning on top-performing product families.")
    print("  2. Schedule additional staffing during payday periods (15th and month-end).")
    print("  3. Plan targeted promotions around holidays to leverage the natural sales uplift.")
    print("  4. Monitor crude oil prices as a major macro-economic leading indicator in Ecuador.")
    print("  5. Factor in external shocks (like the 2016 earthquake) when planning safety stocks.")

    print("\nFUTURE DEVELOPMENT PATHS:")
    print("  - Integrate gradient boosting regressors (XGBoost, LightGBM, CatBoost) into the comparison.")
    print("  - Train individual time-series models for each store-family combination.")
    print("  - Create a weighted ensemble prediction combining Random Forest, Prophet, and TimesFM.")
    print("  - Explore deep learning architectures (LSTMs, Temporal Fusion Transformers).")

    print("\n" + "=" * 80)
    print("Analysis by: Chelsea Ayu")
    print("LinkedIn: https://linkedin.com/in/chelseaayu")
    print("Portfolio: https://chelsea-ayu.vercel.app/")
    print("=" * 80)
