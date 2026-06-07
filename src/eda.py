"""
eda.py — Exploratory Data Analysis: visualisations and summary statistics.

Every plot is saved to ``config.OUTPUT_DIR`` (no interactive windows).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_plot

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def run_eda(
    train: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, pd.Series]:
    """
    Run full EDA and save plots.

    Returns
    -------
    daily_sales, oil_correlation, transaction_correlation, holiday_sales
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── 1. Dataset overview ────────────────────────────────────
    print("\n--- Train Data Overview ---")
    print(train.head())
    print(f"\nDate Range: {train['date'].min()} to {train['date'].max()}")
    print(f"Missing Values:\n{train.isnull().sum()}")

    print("\n--- Stores Data Overview ---")
    print(stores.head())
    print(f"Cities: {stores['city'].nunique()}, States: {stores['state'].nunique()}")
    print(f"Store Types: {stores['type'].unique()}, Clusters: {stores['cluster'].nunique()}")

    print("\n--- Oil Data Overview ---")
    print(oil.head())
    print(f"Missing values: {oil['dcoilwtico'].isnull().sum()}")

    print("\n--- Holidays Data Overview ---")
    print(holidays.head(10))
    print(f"\nHoliday Types:\n{holidays['type'].value_counts()}")

    # ── 2. Sales trend ─────────────────────────────────────────
    daily_sales = train.groupby("date")["sales"].sum().reset_index()

    plt.figure(figsize=(16, 5))
    plt.plot(daily_sales["date"], daily_sales["sales"], linewidth=1)
    plt.title("Daily Total Sales Trend (2013-2017)", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.axvline(
        pd.to_datetime("2016-04-16"), color="red", linestyle="--",
        label="Earthquake (Apr 16, 2016)", linewidth=2,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("01_daily_sales_trend.png")

    print(f"\nTotal Sales: ${daily_sales['sales'].sum():,.2f}")
    print(f"Average Daily Sales: ${daily_sales['sales'].mean():,.2f}")

    # ── 3. Top product families ────────────────────────────────
    family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False)

    plt.figure(figsize=(14, 8))
    family_sales.head(15).plot(kind="barh", color="steelblue")
    plt.title("Top 15 Product Families by Total Sales", fontsize=16, fontweight="bold")
    plt.xlabel("Total Sales")
    plt.tight_layout()
    save_plot("02_top_product_families.png")

    # ── 4. Seasonality ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    train.groupby(train["date"].dt.month)["sales"].mean().plot(
        kind="bar", ax=axes[0], color="steelblue",
    )
    axes[0].set_title("Monthly Average Sales", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average Sales")

    train.groupby(train["date"].dt.dayofweek)["sales"].mean().plot(
        kind="bar", ax=axes[1], color="coral",
    )
    axes[1].set_title("Daily Average Sales (Day of Week)", fontweight="bold")
    axes[1].set_xlabel("Day of Week (0=Mon)")
    axes[1].set_ylabel("Average Sales")

    plt.tight_layout()
    save_plot("03_seasonality.png")

    # ── 5. Oil price ───────────────────────────────────────────
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

    plt.figure(figsize=(16, 5))
    plt.plot(oil["date"], oil["dcoilwtico"], linewidth=1, color="darkgreen")
    plt.title("Daily Oil Price Trend (WTI)", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Oil Price (USD)")
    plt.axvline(
        pd.to_datetime("2016-04-16"), color="red", linestyle="--",
        label="Earthquake", linewidth=2,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("04_oil_price_trend.png")

    daily_sales_oil = daily_sales.merge(oil, on="date", how="left")
    oil_corr = daily_sales_oil["sales"].corr(daily_sales_oil["dcoilwtico"])

    plt.figure(figsize=(10, 6))
    plt.scatter(daily_sales_oil["dcoilwtico"], daily_sales_oil["sales"], alpha=0.5)
    plt.title(f"Oil Price vs Sales (Correlation: {oil_corr:.3f})", fontsize=14, fontweight="bold")
    plt.xlabel("Oil Price (USD)")
    plt.ylabel("Total Daily Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("05_oil_vs_sales.png")

    print(f"\n[INFO] Correlation between Oil Price and Sales: {oil_corr:.4f}")

    # ── 6. Holidays ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    holidays["type"].value_counts().plot(kind="bar", ax=axes[0, 0], color="indianred")
    axes[0, 0].set_title("Holiday Types", fontweight="bold")

    holidays["locale"].value_counts().plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Holiday Locale", fontweight="bold")

    holidays["transferred"].value_counts().plot(kind="bar", ax=axes[1, 0], color="seagreen")
    axes[1, 0].set_title("Transferred Holidays", fontweight="bold")

    holidays["year"] = holidays["date"].dt.year
    holidays.groupby("year").size().plot(kind="bar", ax=axes[1, 1], color="orange")
    axes[1, 1].set_title("Holidays per Year", fontweight="bold")

    plt.tight_layout()
    save_plot("06_holidays_overview.png")

    train["is_holiday"] = train["date"].isin(holidays["date"]).astype(int)
    holiday_sales = train.groupby("is_holiday")["sales"].mean()

    plt.figure(figsize=(10, 6))
    holiday_sales.plot(kind="bar", color=["skyblue", "coral"])
    plt.title("Average Sales: Holiday vs Non-Holiday", fontsize=16, fontweight="bold")
    plt.xlabel("Is Holiday (0=No, 1=Yes)")
    plt.ylabel("Average Sales")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot("07_holiday_vs_nonholiday.png")

    print(f"Average sales on holidays: {holiday_sales[1]:.2f}")
    print(f"Average sales on non-holidays: {holiday_sales[0]:.2f}")
    pct = (holiday_sales[1] - holiday_sales[0]) / holiday_sales[0] * 100
    print(f"Difference: {pct:.2f}%")

    # ── 7. Transactions ────────────────────────────────────────
    daily_tx = transactions.groupby("date")["transactions"].sum().reset_index()

    plt.figure(figsize=(16, 5))
    plt.plot(daily_tx["date"], daily_tx["transactions"], linewidth=1, color="purple")
    plt.title("Daily Total Transactions", fontsize=16, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Transactions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("08_daily_transactions.png")

    sales_by_sd = train.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
    tx_sales = transactions.merge(sales_by_sd, on=["date", "store_nbr"], how="inner")
    tx_corr = tx_sales["transactions"].corr(tx_sales["sales"])

    plt.figure(figsize=(10, 6))
    plt.scatter(tx_sales["transactions"], tx_sales["sales"], alpha=0.3)
    plt.title(
        f"Transactions vs Sales (Correlation: {tx_corr:.3f})",
        fontsize=14, fontweight="bold",
    )
    plt.xlabel("Transactions")
    plt.ylabel("Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("09_transactions_vs_sales.png")

    print(f"\n[INFO] Correlation between Transactions and Sales: {tx_corr:.4f}")

    return daily_sales, oil_corr, tx_corr, holiday_sales
