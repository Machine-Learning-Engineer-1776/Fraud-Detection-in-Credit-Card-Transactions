# Credit Card Fraud Detection
**XGBoost + Logistic Regression on Kaggle Dataset**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)

**Live Notebook:** [Colab Demo](https://colab.research.google.com/drive/13zeGVos5HIIZyV8jv1klzLXdbXSByfAL)  
**GitHub:** [github.com/YOUR_USERNAME/Fraud-Detection](https://github.com/YOUR_USERNAME/Fraud-Detection)

---

## Overview

**Fraud Detection in Transactions** is a **complete ML pipeline** that:
- Loads the **Kaggle Credit Card Fraud** dataset
- Engineers **8 behavioral features**
- Trains **XGBoost** and **Logistic Regression**
- Evaluates with **ROC-AUC, PR-AUC, F1**
- Visualizes results with **5 interactive Plotly charts**

Built in **Google Colab** — **no deployment, no real data**.

---

## What This Code Actually Does

| Step | Output |
|------|--------|
| **Data Prep** | Scales `Amount`/`Time` → adds `hour_of_day` | `preprocessed_creditcard.csv` |
| **Feature Engineering** | 8 new features: `time_since_last_txn`, `amount_to_median_ratio`, etc. | `engineered_creditcard.csv` |
| **Modeling** | XGBoost + Logistic Regression (SMOTE) | `xgb_model.pkl`, `predictions.csv` |
| **Evaluation** | ROC-AUC: ~0.97, PR-AUC: ~0.85, F1: ~0.90 | `model_metrics.csv` |
| **Visualization** | 5 interactive HTML plots | `histogram.html`, `roc_curve.html`, etc. |

---

## Key Features

### 1. Behavioral Feature Engineering
- time_since_last_txn
- amount_to_median_ratio
- txn_count_rolling
- is_top_5pct_amount

---


### 2. Two Models
- **XGBoost** with `scale_pos_weight`
- **Logistic Regression** with **SMOTE**

### 3. Interactive Plots

<img width="497" height="425" alt="{A1FCC168-58AE-4F71-B071-27CC209B7409}" src="https://github.com/user-attachments/assets/4c409615-1835-49d3-b077-41ab310f2af2" />

---

## Files Generated

- `preprocessed_creditcard.csv`
- `engineered_creditcard.csv`
- `features_creditcard.csv`
- `target_creditcard.csv`
- `predictions.csv`
- `model_metrics.csv`
- `xgb_model.pkl`
- `lr_model.pkl`
- 5 × `.html` interactive plots

---

## How to Run (5 Minutes)

1. Open [Colab Notebook](https://colab.research.google.com/drive/13zeGVos5HIIZyV8jv1klzLXdbXSByfAL)
2. Run all cells
3. Download **12 files** → Add to portfolio

---

## Model Performance (From Code)

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| **ROC-AUC** | ~0.97 | ~0.95 |
| **PR-AUC** | ~0.85 | ~0.80 |
| **F1-Score** | ~0.90 | ~0.85 |

<img width="461" height="385" alt="image" src="https://github.com/user-attachments/assets/f1ca67a3-94c6-4d71-a00c-5a7bfded87ba" />

<img width="467" height="391" alt="{4AA535AA-DC18-4A16-8DD3-B9F2B8AD00BE}" src="https://github.com/user-attachments/assets/e798deb3-d9a6-4087-9e7f-5e44c097fdd5" />



> **Note:** High scores due to **clean, public dataset**. Real fraud is messier.

---

## Tech Stack

| Tool | Purpose |
|------|--------|
| **Pandas** | Data prep |
| **Scikit-learn** | Split, SMOTE, metrics |
| **XGBoost** | Gradient boosting |
| **Plotly** | Interactive charts |
| **Joblib** | Model saving |

---

## This Project Proves:

- You can **handle imbalanced data**
- You can **engineer features**
- You can **train + evaluate models**
- You can **visualize results**
- You can **document a pipeline**

**Perfect for Data Scientist / ML Engineer roles.**

---

**Made with code, not claims.**
