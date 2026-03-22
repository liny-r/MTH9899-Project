# MTH9899 Final Project — Next-Day Residual Return Prediction

A machine learning pipeline to predict next-day residual stock returns at 15:30, trained on ~500 US equities over 2010–2014.

## Project Overview

**Prediction task:** At 15:30 each trading day, predict the 24-hour ahead residual return for each stock in the universe.

**Target variable:** `Part A (15:30→16:00 same day) + Part B (open→15:30 next day)` residual returns, vol-scaled and normalized.

**Train:** 2010–2013 (~1.26M rows) | **Validation:** 2014 (~315K rows)

**Best model:** XGBoost (selected by liquidity-weighted R² on validation set)

---

## Repository Structure

```
MTH9899-Project/
├── src/
│   ├── data.py         # Data loading and date map utilities
│   ├── target.py       # Target variable construction pipeline
│   ├── features.py     # Feature engineering (intraday + daily)
│   ├── train.py        # Model training and hyperparameter search
│   ├── predict.py      # Inference with saved artifacts
│   └── utils.py        # Shared utilities (weighted R², etc.)
├── tests/
│   └── test_features.py
├── saved_model/
│   ├── best_model.pkl      # Fitted XGBoost model
│   ├── feature_cols.pkl    # Ordered list of 13 feature names
│   ├── scaler.pkl          # StandardScaler fitted on train set
│   └── fit_target_mode.pkl # Target normalization mode ('vol_scaled')
├── run.ipynb               # Orchestration notebook (full pipeline)
├── Full_Pipeline.ipynb     # Legacy monolithic notebook
├── requirements.txt
└── CLAUDE.md
```

Data directories (gitignored):
```
DailyData/          # dat.YYYYMMDD.csv — OHLCV + adjustment factors
data_intraday/      # YYYYMMDD.csv — 15-min residual return snapshots
```

---

## Setup

This project runs in a Docker DevContainer with Python 3.

```bash
pip install -r requirements.txt
```

Required packages: `numpy pandas scikit-learn xgboost joblib scipy statsmodels matplotlib plotly tqdm python-dotenv pytest`

---

## Running the Pipeline

Open and run `run.ipynb` from the `MTH9899-Project/` directory. The notebook steps through:

1. Load date maps and daily data
2. Build the target variable
3. Train/validation split (2010–2013 train, 2014 val)
4. Build and normalize features
5. Apply target normalization pipeline
6. Train all five models with hyperparameter search
7. Select and save the best model
8. Feature importance (permutation MDA)
9. Bin plots and white paper visualizations

```bash
jupyter notebook run.ipynb
```

Run tests:
```bash
pytest
```

---

## Features (13 total)

All features use only data available at or before 15:30 on the prediction day. Each feature goes through: per-Id time-series z-score (252-day rolling, past-only) → cross-sectional ±5 MAD winsorization → cross-sectional z-score.

### Intraday Features (from `data_intraday/`)

| Feature | Definition |
|---|---|
| `OvernightReturn` | `CumReturnResid` at 09:45 |
| `FirstHourMomentum` | `CumReturnResid(12:00) − CumReturnResid(09:45)` |
| `LastHourMomentum` | `CumReturnResid(15:30) − CumReturnResid(14:30)` |
| `IntradayReversal` | Morning return − afternoon return |
| `IntradayReturnSkew` | Skewness of 15-min residual return increments |
| `RealizedUpsideVol` | RMS of positive 15-min increments |
| `VolumeSurprise` | `CumVolume(15:30) / MDV_63_prev` |
| `VolumeMorningAfternoonRatio` | Morning volume / afternoon volume (split at 12:00) |
| `RetVolCorr` | Pearson correlation of 15-min return and volume increments |

### Daily Features (previous trading day — no look-ahead)

| Feature | Definition |
|---|---|
| `VolatilityAdjustedReturn` | `CumReturnResid(15:30) / EST_VOL_prev` |
| `ShortTermReversal` | `Close_adj(t−1) / Close_adj(t−2) − 1` |
| `Momentum21d` | `Close_adj(t−2) / Close_adj(t−23) − 1` (skips t−1) |
| `DollarVolTrend` | `mean(DollarVol t−1..t−5) / MDV_63_prev` |

---

## Models

Five models trained; best selected by validation liquidity-weighted R² (weights = `sqrt(MDV_63_prev)`):

| Model | Hyperparameter Search |
|---|---|
| Ridge | alpha ∈ logspace(−3, 3, 25) |
| Random Forest | n_estimators ∈ {80,150}, max_depth ∈ {5,7,10}, min_samples_leaf ∈ {10,20} |
| **XGBoost** ✓ | max_depth ∈ {3–6}, lr ∈ {0.03,0.05,0.1}, n_estimators ∈ {100,200,300} |
| ElasticNet | alpha ∈ logspace(−5, −0.5, 12), l1_ratio ∈ {0.1,0.5,0.8,0.9,0.95,1.0} |
| MLP | hidden layers ∈ {(64,),(128,),(128,64)}, alpha ∈ {1e-4,1e-3,1e-2}, lr ∈ {1e-3,2e-3} |

---

## Inference

```python
from src.predict import predict
import pandas as pd

# df must contain the 13 feature columns (already normalized)
preds = predict(df, model_dir='saved_model')
```

Or load artifacts manually:

```python
import pickle

with open('saved_model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('saved_model/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('saved_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = scaler.transform(df[feature_cols].astype(float))
preds = model.predict(X)
```

---

## Data Format

### `DailyData/dat.YYYYMMDD.csv`
`Date, ID, SYMBOL, MIC, FREE_FLOAT_PERCENTAGE, EST_VOL, MDV_63, Open, High, Low, Close, Volume, PxAdjFactor, SharesAdjFactor`

- Adjusted close: `Close_adj = Close × PxAdjFactor`
- `MDV_63`: 63-day median daily dollar volume
- `EST_VOL`: annualized volatility estimate

### `data_intraday/YYYYMMDD.csv`
`Date, Time, Id, CumReturnResid, CumReturnRaw, CumVolume`

- 15-minute snapshots from 09:45; key timestamps: 09:45, 12:00, 14:30, 15:30
