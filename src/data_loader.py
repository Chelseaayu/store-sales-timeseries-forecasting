"""
data_loader.py — Load all CSV datasets from local disk.
"""

import os
import pandas as pd

from config import DATA_DIR


def load_data(data_path: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load every CSV file needed for the project.

    Parameters
    ----------
    data_path : str, optional
        Folder containing the CSV files.  Defaults to the project root
        (``config.DATA_DIR``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: train, test, stores, oil, holidays, transactions.
    """
    path = data_path or DATA_DIR
    path = path if path.endswith(os.sep) else path + os.sep

    print("Loading datasets...")
    print(f"Reading from: {path}\n")

    data = {
        "train": pd.read_csv(path + "train.csv", parse_dates=["date"]),
        "test": pd.read_csv(path + "test.csv", parse_dates=["date"]),
        "stores": pd.read_csv(path + "stores.csv"),
        "oil": pd.read_csv(path + "oil.csv", parse_dates=["date"]),
        "holidays": pd.read_csv(
            path + "holidays_events.csv", parse_dates=["date"]
        ),
        "transactions": pd.read_csv(
            path + "transactions.csv", parse_dates=["date"]
        ),
    }

    print("[SUCCESS] All datasets loaded successfully!")
    print("\nDataset Shapes:")
    for name, df in data.items():
        print(f"  {name:>12}: {df.shape}")

    return data
