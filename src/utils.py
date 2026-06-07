"""
utils.py — Shared helper functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure to OUTPUT_DIR and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [INFO] Plot saved: {filepath}")


def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error (Kaggle competition metric)."""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
