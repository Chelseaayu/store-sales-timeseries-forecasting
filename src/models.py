"""
models.py — Training functions for Baseline, Random Forest, Prophet, and TimesFM.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF,
    RF_MAX_TRAIN_SAMPLES,
    RF_RANDOM_STATE,
    TIMESFM_BATCH_SIZE,
    TIMESFM_HORIZON,
    TIMESFM_CHECKPOINT,
)

# Suppress warnings from Prophet and PyStan/CmdStanPy
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Graceful imports for optional advanced models
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("[WARN] Prophet is not installed. Prophet training will be skipped.")

HAS_TIMESFM = False
TIMESFM_VERSION = None

try:
    import timesfm
    # Detect TimesFM version API (TimesFM 2.5 has TimesFM_2p5_200M_torch class)
    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        TIMESFM_VERSION = "2.5"
        HAS_TIMESFM = True
        print("[INFO] Detected TimesFM version 2.5+ API.")
    elif hasattr(timesfm, "TimesFm") or hasattr(timesfm, "timesfm"):
        TIMESFM_VERSION = "1.0"
        HAS_TIMESFM = True
        print("[INFO] Detected TimesFM version 1.0 API.")
    else:
        # Fallback check
        try:
            from timesfm import TimesFm
            TIMESFM_VERSION = "1.0"
            HAS_TIMESFM = True
            print("[INFO] Detected TimesFM version 1.0 API (via fallback import).")
        except ImportError:
            pass
except (ImportError, ModuleNotFoundError) as e:
    pass

if not HAS_TIMESFM:
    print("[WARN] TimesFM is not installed or failed to import dependencies (like PyTorch). TimesFM forecasting will be skipped.")


def train_baseline(y_train: pd.Series, y_val: pd.Series) -> tuple[np.ndarray, float, float]:
    """
    Baseline model: Predicts the mean of the training sales for all validation dates.
    
    Returns
    -------
    y_pred, rmse, mae
    """
    print("\n" + "=" * 60)
    print("BASELINE MODEL (Mean Prediction)")
    print("=" * 60)

    mean_sales = y_train.mean()
    baseline_pred = np.full(len(y_val), mean_sales)
    rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
    mae = mean_absolute_error(y_val, baseline_pred)

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")
    return baseline_pred, float(rmse), float(mae)


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[RandomForestRegressor, np.ndarray, float, float]:
    """
    Train a Random Forest model on a subset of the training data.
    
    Returns
    -------
    rf_model, y_pred, rmse, mae
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)
    print("Training Random Forest model...")
    print(f"(Using a subset of up to {RF_MAX_TRAIN_SAMPLES:,} samples for faster training)")

    sample_size = min(RF_MAX_TRAIN_SAMPLES, len(X_train))
    if sample_size < len(X_train):
        sample_indices = np.random.choice(X_train.index, size=sample_size, replace=False)
        X_train_sample = X_train.loc[sample_indices]
        y_train_sample = y_train.loc[sample_indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    rf_model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    rf_model.fit(X_train_sample, y_train_sample)

    rf_pred_val = np.maximum(rf_model.predict(X_val), 0)
    rmse = np.sqrt(mean_squared_error(y_val, rf_pred_val))
    mae = mean_absolute_error(y_val, rf_pred_val)

    print(f"\n[SUCCESS] Random Forest training completed!")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")

    return rf_model, rf_pred_val, float(rmse), float(mae)


def train_prophet(
    train: pd.DataFrame, split_date: pd.Timestamp, top_combinations: list[tuple[int, str]]
) -> tuple[dict, pd.DataFrame, float, float]:
    """
    Train Prophet models on top store-family combinations.
    
    Returns
    -------
    prophet_models_dict, prophet_val_df, rmse, mae
    """
    print("\n" + "=" * 60)
    print("PROPHET MODEL")
    print("=" * 60)

    if not HAS_PROPHET:
        print("Prophet is not available. Skipping Prophet training.")
        return {}, pd.DataFrame(), 0.0, 0.0

    print(f"Training Prophet models on top {len(top_combinations)} store-family combinations...")

    train_prophet_df = train[["date", "store_nbr", "family", "sales"]].copy()
    train_prophet_df = train_prophet_df.rename(columns={"date": "ds", "sales": "y"})

    prophet_models = {}
    prophet_val_predictions = []

    for store_nbr, family in top_combinations:
        subset = train_prophet_df[
            (train_prophet_df["store_nbr"] == store_nbr) &
            (train_prophet_df["family"] == family)
        ][["ds", "y"]].copy()

        subset_train = subset[subset["ds"] <= split_date]
        subset_val = subset[subset["ds"] > split_date]

        if len(subset_train) < 30 or len(subset_val) == 0:
            continue

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(subset_train)

        future_val = subset_val[["ds"]].copy()
        forecast_val = model.predict(future_val)

        preds = forecast_val[["ds", "yhat"]].copy()
        preds["yhat"] = np.maximum(preds["yhat"], 0)
        preds["store_nbr"] = store_nbr
        preds["family"] = family
        preds["actual"] = subset_val["y"].values[: len(preds)]
        prophet_val_predictions.append(preds)
        prophet_models[(store_nbr, family)] = model

    if not prophet_val_predictions:
        print("[WARN] No valid data for Prophet training on the specified combinations.")
        return {}, pd.DataFrame(), 0.0, 0.0

    prophet_val_df = pd.concat(prophet_val_predictions, ignore_index=True)
    rmse = np.sqrt(mean_squared_error(prophet_val_df["actual"], prophet_val_df["yhat"]))
    mae = mean_absolute_error(prophet_val_df["actual"], prophet_val_df["yhat"])

    print(f"\n[SUCCESS] Prophet training completed!")
    print(f"Models trained: {len(prophet_models)}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")

    return prophet_models, prophet_val_df, float(rmse), float(mae)


def train_timesfm(
    train: pd.DataFrame, split_date: pd.Timestamp, top_combinations: list[tuple[int, str]]
) -> tuple[pd.DataFrame, float, float]:
    """
    Run the pre-trained TimesFM foundation model on top store-family combinations.
    Supports both TimesFM 1.0 and TimesFM 2.5 APIs automatically.
    
    Returns
    -------
    timesfm_val_df, rmse, mae
    """
    print("\n" + "=" * 60)
    print("TIMESFM MODEL")
    print("=" * 60)

    if not HAS_TIMESFM:
        print("TimesFM is not available. Skipping TimesFM forecasting.")
        return pd.DataFrame(), 0.0, 0.0

    print(f"Loading pre-trained TimesFM (API version {TIMESFM_VERSION})...")
    
    try:
        if TIMESFM_VERSION == "2.5":
            # TimesFM 2.5 API initialization
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch",
                torch_compile=False
            )
            # Create ForecastConfig and compile model
            fc = timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=TIMESFM_HORIZON,
                per_core_batch_size=TIMESFM_BATCH_SIZE,
            )
            tfm.compile(fc)
        else:
            # TimesFM 1.0 API initialization
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",
                    per_core_batch_size=TIMESFM_BATCH_SIZE,
                    horizon_len=TIMESFM_HORIZON,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=TIMESFM_CHECKPOINT
                ),
            )
    except Exception as e:
        print(f"[ERROR] Failed to load TimesFM model checkpoints: {e}")
        print("Please check if you have a valid internet connection or local HuggingFace cache.")
        return pd.DataFrame(), 0.0, 0.0

    timesfm_val_predictions = []

    for store_nbr, family in top_combinations:
        subset = train[["date", "store_nbr", "family", "sales"]].copy()
        subset = subset[
            (subset["store_nbr"] == store_nbr) & (subset["family"] == family)
        ].sort_values("date")

        subset_train = subset[subset["date"] <= split_date]
        subset_val = subset[subset["date"] > split_date]

        if len(subset_train) < 30 or len(subset_val) == 0:
            continue

        history = subset_train["sales"].values.astype(np.float32)
        forecast_length = min(len(subset_val), TIMESFM_HORIZON)
        
        try:
            if TIMESFM_VERSION == "2.5":
                # TimesFM 2.5 forecast method
                point_forecast, _ = tfm.forecast(TIMESFM_HORIZON, [history])
            else:
                # TimesFM 1.0 forecast method
                point_forecast, _ = tfm.forecast([history], freq=[1])
        except Exception as e:
            print(f"[ERROR] Forecast execution failed for store {store_nbr} - {family}: {e}")
            continue

        preds = np.maximum(point_forecast[0][:forecast_length], 0)
        actuals = subset_val["sales"].values[:forecast_length]
        dates = subset_val["date"].values[:forecast_length]

        pred_df = pd.DataFrame({
            "ds": dates,
            "yhat": preds,
            "store_nbr": store_nbr,
            "family": family,
            "actual": actuals,
        })
        timesfm_val_predictions.append(pred_df)

    if not timesfm_val_predictions:
        print("[WARN] No valid data for TimesFM forecasting on the specified combinations.")
        return pd.DataFrame(), 0.0, 0.0

    timesfm_val_df = pd.concat(timesfm_val_predictions, ignore_index=True)
    rmse = np.sqrt(mean_squared_error(timesfm_val_df["actual"], timesfm_val_df["yhat"]))
    mae = mean_absolute_error(timesfm_val_df["actual"], timesfm_val_df["yhat"])

    print(f"\n[SUCCESS] TimesFM forecasting completed!")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")

    return timesfm_val_df, float(rmse), float(mae)
