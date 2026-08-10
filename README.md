# Mutual Fund Risk Classification & Portfolio Recommendation System

A machine learning pipeline that predicts a mutual fund's risk category (Low / Medium / High) from its everyday characteristics, and uses that model to power a fund clustering and recommendation system.

## Overview

Mutual fund investors are usually shown a risk rating built largely from volatility statistics (standard deviation, beta, Sharpe ratio, etc.). This project asks: **can that risk category be predicted just as well from a fund's basic characteristics** — cost, size, age, category, fund house — **without leaning on the volatility metrics that typically define it?**

To answer this honestly rather than assuming, two XGBoost models are trained and compared side by side:

- **Full model** — all engineered features, including volatility metrics (19 features)
- **Fundamental-only model** — the same features, minus the volatility metrics (14 features)

**Result:** the fundamental-only model matches — and marginally exceeds — the full model (89.6% vs 89.0% test accuracy), showing that a fund's basic characteristics carry real predictive signal on their own.

## What's in this project

- **Data cleaning** — handling `"-"` placeholder values and missing data via sub-category-wise median imputation
- **Exploratory Data Analysis** — risk-vs-return patterns, a correlation study motivating the feature comparison below, cost differences by fund category, and dataset composition
- **Feature engineering** — categorical encoding, a fund-manager workload proxy, and a cost-efficiency metric
- **Feature scaling** — standardization applied across the pipeline
- **A direct comparison of two feature sets** to test whether volatility features are actually necessary for good accuracy
- **XGBoost classification** with full evaluation (accuracy, per-class precision/recall/F1, confusion matrix, feature importance)
- **KMeans + PCA clustering** to segment funds by overall similarity
- **A weighted, multi-criteria recommendation engine** that ranks funds for an investor by risk preference, category, and monthly SIP budget

## Dataset

`comprehensive_mutual_funds_data.csv` — 814 Indian mutual funds, 20 columns, including fund identity, cost, size, historical returns, risk/volatility statistics, and a star rating.

## Results

| Model | Features | Accuracy | Weighted F1 |
|---|---|---|---|
| Full (incl. volatility features) | 19 | 0.890 | 0.89 |
| **Fundamental-only** | **14** | **0.896** | **0.90** |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| High | 0.98 | 0.99 | 0.98 |
| Low | 0.78 | 0.86 | 0.82 |
| Medium | 0.82 | 0.71 | 0.76 |

## Tech Stack

- **Python** — pandas, numpy
- **Modeling** — scikit-learn, XGBoost
- **Visualization** — matplotlib, seaborn

## Author

**Prakhar Gupta**
M.Sc. Statistics And Computing , 2nd Year
