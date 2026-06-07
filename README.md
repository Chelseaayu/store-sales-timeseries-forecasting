# 🏪 Store Sales Time Series Forecasting

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Predicting sales for Corporación Favorita stores in Ecuador using modular machine learning, time-series statistical models, and modern foundation models (TimesFM).**

---

## 👩‍💻 Author

**Chelsea Ayu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Chelsea_Ayu-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/chelseaayu)
[![Portfolio](https://img.shields.io/badge/Portfolio-chelsea--ayu.vercel.app-000000?style=flat&logo=vercel&logoColor=white)](https://chelsea-ayu.vercel.app/)

---

## 📊 Project Overview

This project tackles the **Store Sales Time Series Forecasting** challenge from Kaggle, where the objective is to predict the unit sales for thousands of product families sold at Corporación Favorita stores located in Ecuador. 

Predicting store sales accurately is vital to inventory optimization, staffing schedules, and promotion strategies. To solve this, we implement a modular data science pipeline exploring classical machine learning, statistical models, and deep-learning foundation models.

---

## 📁 Repository Structure

This repository is designed with **clean code** and **modular software engineering** best practices, making it production-ready and easy to maintain compared to standard monolothic Jupyter notebooks.

```
store-sales-time-series-forecasting/
│
├── data/                            # Folder containing CSV datasets
│   ├── train.csv (IGNORED)          # Historical training set (>100MB, excluded from Git)
│   ├── test.csv                     # Test set for prediction
│   ├── stores.csv                   # Store metadata
│   ├── oil.csv                      # Daily oil prices
│   ├── holidays_events.csv          # Holiday calendar
│   ├── transactions.csv             # Store transaction history
│   └── sample_submission.csv        # Kaggle submission template
│
├── src/                             # Python source modules
│   ├── config.py                    # Constants, paths, and hyperparameters
│   ├── data_loader.py               # Local data loading utilities
│   ├── eda.py                       # Exploratory Data Analysis & plot generator
│   ├── preprocessing.py             # Feature engineering and data split logic
│   ├── utils.py                     # Metric calculations and plotting helpers
│   ├── models.py                    # Regressors: Baseline, Random Forest, Prophet, TimesFM
│   ├── evaluation.py                # Validation metrics comparison and graphing
│   ├── predict.py                   # Prediction generation on the test set
│   └── main.py                      # Main entry point and pipeline orchestrator
│
├── outputs/                         # Folder containing saved plot images
├── .gitignore                       # Git exclusion rules
├── LICENSE                          # MIT License file
├── requirements.txt                 # Dependencies list
├── submission.csv                   # Generated prediction file (excluded from Git)
```

---

## 🔍 Feature Engineering Pipeline

We engineered over **30+ advanced features** to capture seasonality, macro-economic factors, and external shocks:

*   📅 **Temporal Features**: Day, Month, Year, Day of Week, Quarter, Weekend indicators.
*   💰 **Payday Indicators**: Flags matching Ecuador's bi-monthly paydays (15th and month-end) and a proximity-to-payday count.
*   🌍 **External Shocks**: April 16, 2016 Ecuador earthquake impact analysis (days elapsed since the earthquake).
*   🛢️ **Macro-economic factors**: Daily crude oil price (WTI) combined with 7-day and 30-day moving averages (essential since Ecuador is highly dependent on oil exports).
*   🎉 **Holidays**: Separated by National, Regional, and Transferred holiday classifications.
*   📈 **Time-Series Lags & Rolling Metrics**: 7-day, 14-day, and 30-day historical sales lags, alongside rolling mean and standard deviation sales metrics.
*   🏷️ **Categorical Encoding**: Robust encoding of categorical features (Family, City, State, Store Type).

---

## 🤖 Models Implemented

We evaluate four time series modeling methodologies, progressing from simple baselines to state-of-the-art foundation models:

1.  **Baseline (Mean Prediction)**: Predicts the rolling historical mean sales. Serves as a reference.
2.  **Random Forest Regressor (Scikit-Learn)**: Trained using feature matrices containing lagged sales, temporal, holiday, and oil price features.
3.  **Prophet (Meta)**: An additive regression model designed for analyzing weekly and yearly seasonality alongside holiday effects.
4.  **TimesFM (Google)**: A 200M parameter decoder-only foundation model pre-trained on billions of real-world time-series data points, utilized zero-shot.

---

## 🏆 Model Performance Comparison

The models are compared on a 30-day validation set using **Root Mean Squared Logarithmic Error (RMSLE)**, which is the official competition metric on Kaggle, alongside **RMSE** and **MAE**:

| Rank | Model | RMSE | MAE | RMSLE |
| :--- | :--- | :--- | :--- | :--- |
| **1** 🥇 | **TimesFM (Google 2.5)** | **960.94** | **730.37** | **0.0926** |
| **2** | **Random Forest** | *1145.71* | *902.96* | **0.1175** |
| **3** | **Prophet (Meta)** | *1537.65* | *1181.18* | **0.1583** |
| **4** | **Baseline (Mean)** | *9227.37* | *8919.47* | **3.2373** |

*Note: Models are evaluated on the top 10 highest-volume store-family combinations. Google's TimesFM 2.5 foundation model achieved the best results (RMSLE: 0.0926), demonstrating the power of zero-shot time-series foundation models over traditional machine learning and statistical models on this dataset.*

---

## 💡 Key Business Insights

1.  **Macro-Economic Sensitivity**: Sales exhibit a moderate negative correlation with crude oil prices. As oil prices drop, purchasing power shifts, impacting discretionary spend.
2.  **The Payday Effect**: Dramatic spikes in consumer purchase volume occur on the **15th** and **30th/31st** of each month.
3.  **Holiday Elasticity**: On average, sales increase by **~12-15%** on holiday dates, indicating key windows for promotional campaigns.
4.  **Disaster Resilience**: Analysis of the 2016 earthquake shows a 3-week anomaly where basic goods surged in volume while general retail sales dipped.

---

## 🚀 Getting Started

### 📋 Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/store-sales-time-series-forecasting.git
   cd store-sales-time-series-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `timesfm` is optional and requires PyTorch. If it is not present, the pipeline will skip TimesFM and run the rest of the models smoothly.)*

### 📥 Downloading the Data

> [!IMPORTANT]
> Because **`train.csv` exceeds 100MB**, it is ignored in `.gitignore` to comply with GitHub's file size limits.
> You must download the datasets from Kaggle to run this repository.

1. Go to the [Kaggle Store Sales Competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
2. Download the data files and place the CSV files directly in the `data/` folder of this repository:
   - `train.csv`
   - `test.csv`
   - `stores.csv`
   - `oil.csv`
   - `holidays_events.csv`
   - `transactions.csv`
   - `sample_submission.csv`

### 💻 Running the Pipeline

You can run the entire pipeline—from raw data loading and EDA to training, evaluation, and test predictions—with a single command:

```bash
python src/main.py
```

#### CLI Command Options:
*   **Skip EDA generation** (saves execution time by skipping plot drawing):
    ```bash
    python src/main.py --skip-eda
    ```
*   **Skip slow models** (Prophet and TimesFM require higher CPU/GPU power. Use this flag to run only Random Forest and Baseline):
    ```bash
    python src/main.py --skip-prophet --skip-timesfm
    ```
*   **Run a customized path for CSV files**:
    ```bash
    python src/main.py --data-path "C:/path/to/datasets/"
    ```

---

## 🤝 Connect & Collaborate

If you have feedback, questions, or ideas for improving the forecasting pipeline, let's connect!

*   💼 LinkedIn: [linkedin.com/in/chelseaayu](https://linkedin.com/in/chelseaayu)
*   🌐 Portfolio: [chelsea-ayu.vercel.app](https://chelsea-ayu.vercel.app/)

---
<div align="center">
  <p><strong>Made with ❤️ by Chelsea Ayu</strong></p>
  <p>📊 Data Science | 🤖 Machine Learning | 📈 Time Series Forecasting</p>
</div>
