"""
predict.py — Generate predictions for the test dataset and format submission file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FEATURE_COLS, DATA_DIR, PROJECT_ROOT
from utils import save_plot


def generate_submission(rf_model, test: pd.DataFrame, data_path: str | None = None) -> pd.DataFrame:
    """
    Generate test predictions using the trained Random Forest model and write to submission.csv.
    
    Parameters
    ----------
    rf_model : sklearn Regressor
        The trained regression model.
    test : pd.DataFrame
        Preprocessed test dataset containing features.
    data_path : str, optional
        Folder containing the sample_submission.csv file.
    
    Returns
    -------
    pd.DataFrame
        Submission DataFrame containing 'id' and 'sales'.
    """
    print("\n" + "=" * 60)
    print("GENERATE PREDICTIONS FOR TEST SET")
    print("=" * 60)

    # Fill any NaNs in the test feature columns
    X_test = test[FEATURE_COLS].fillna(test[FEATURE_COLS].median())

    print(f"Model used: Random Forest (full coverage for all {len(test):,} test samples)")
    test_predictions = np.maximum(rf_model.predict(X_test), 0)

    print(f"\n[SUCCESS] Predictions generated for {len(test_predictions):,} samples")
    print(f"  - Min  : {test_predictions.min():.2f}")
    print(f"  - Max  : {test_predictions.max():.2f}")
    print(f"  - Mean : {test_predictions.mean():.2f}")
    print(f"  - Median: {np.median(test_predictions):.2f}")

    # Read sample submission and fill predictions
    path = data_path or DATA_DIR
    path = path if path.endswith(os.sep) else path + os.sep
    sample_sub_path = os.path.join(path, "sample_submission.csv")

    if os.path.exists(sample_sub_path):
        submission = pd.read_csv(sample_sub_path)
        submission["sales"] = test_predictions
    else:
        print("[WARN] sample_submission.csv not found in data folder. Creating submission from test data IDs...")
        submission = pd.DataFrame({
            "id": test["id"],
            "sales": test_predictions
        })

    submission_out_path = os.path.join(PROJECT_ROOT, "submission.csv")
    submission.to_csv(submission_out_path, index=False)

    print(f"\n[SUCCESS] Submission file created successfully at: {submission_out_path}")
    print(submission.head(10))

    # Visualize predictions
    test_viz = test.copy()
    test_viz["predictions"] = test_predictions
    daily_pred = test_viz.groupby("date")["predictions"].sum().reset_index()

    plt.figure(figsize=(16, 5))
    plt.plot(daily_pred["date"], daily_pred["predictions"], linewidth=2, marker="o", color="dodgerblue")
    plt.title("Predicted Sales for Next 15 Days (Aggregated)", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Predicted Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("14_test_predictions.png")

    return submission
