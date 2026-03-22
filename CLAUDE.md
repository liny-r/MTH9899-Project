# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project runs in a Docker DevContainer. The container provides Python 3 with: numpy, pandas, jupyter, statsmodels, scipy, scikit-learn, plotly, joblib, tqdm, python-dotenv, pytest.

- Start Jupyter: `jupyter notebook` or `jupyter lab`
- Run tests: `pytest`

## Project Goal

Build a **machine learning model to predict next-day residual stock returns at 15:30**, trained on 5 years of data (2010–2014) for ~500 stocks. The project spec is in `project_assignment.pdf`; the team's feature and model plan is in `Immediate Deliverables.txt`.

## Target Variable

Predict the return over the next 24 hours, constructed as:
- **Part A:** Same-day 15:30→16:00 residual return = `CumReturnResid(16:00) − CumReturnResid(15:30)` on day D
- **Part B:** Next trading day: `CumReturnResid(15:30)` on day D+1
- `target = Part A + Part B`

**Fitting label pipeline (`fit_target_mode = 'vol_scaled'`):**
1. Normalize by previous day's `EST_VOL`
2. Per-Id time-series z-score (252-day rolling window, min_periods=60, past-only — no look-ahead)
3. Cross-sectional ±5 MAD winsorization per date
4. Cross-sectional z-score per date

**Sample weights:** `sqrt(MDV_63_prev)` for liquidity-weighted R² evaluation.

Train: 2010–2013 (~1.26M rows). Validation: 2014 (~315K rows).

## Data

Two datasets, one CSV per trading day, covering **2010-01-04 to 2014-12-31**, ~1,258 securities per day identified by Bloomberg ID.

### DailyData/ — `dat.YYYYMMDD.csv`
`Date, ID, SYMBOL, MIC, FREE_FLOAT_PERCENTAGE, EST_VOL, MDV_63, Open, High, Low, Close, Volume, PxAdjFactor, SharesAdjFactor`

- Prices must be adjusted: `Close_adj = Close * PxAdjFactor`
- `MDV_63`: 63-day median daily dollar volume (liquidity baseline)
- `EST_VOL`: annualized volatility estimate (used for normalization throughout)

### data_intraday/ — `YYYYMMDD.csv`
`Date, Time, Id, CumReturnResid, CumReturnRaw, CumVolume`

- Sampled at 15-minute intervals starting 09:45; key timestamps: `09:45`, `12:00`, `14:30`, `15:30`
- All features must use only data available at or before 15:30

### Held-out test evaluation sample
- this exists, but I've not downloaded it yet. We will run it and build a report after learning about the current work and making sure it runs. 

## Feature Engineering

**13 implemented features** (saved in `saved_model/feature_cols.pkl`). Normalization pipeline per feature: TS z-score (per-Id, 252-day rolling, past-only) → ±5 MAD cross-sectional winsorization → cross-sectional z-score. Volume/ratio features use 1st–99th percentile clipping instead of ±5 MAD.

**Intraday features (from data_intraday/, all at or before 15:30):**
| Feature | Definition |
|---|---|
| `OvernightReturn` | `CumReturnResid` at 09:45 |
| `FirstHourMomentum` | `CumReturnResid(12:00) − CumReturnResid(09:45)` |
| `LastHourMomentum` | `CumReturnResid(15:30) − CumReturnResid(14:30)` |
| `IntradayReversal` | Morning return − afternoon return (09:45→12:00 minus 12:00→15:30) |
| `IntradayReturnSkew` | Skewness of 15-min residual return increments (09:45–15:30) |
| `RealizedUpsideVol` | RMS of positive 15-min increments |
| `VolumeSurprise` | `CumVolume(15:30) / MDV_63_prev` |
| `VolumeMorningAfternoonRatio` | Morning vol / afternoon vol (split at 12:00) |
| `RetVolCorr` | Pearson corr of 15-min return and volume increments |

**Daily features (using previous trading day's data — no look-ahead):**
| Feature | Definition |
|---|---|
| `VolatilityAdjustedReturn` | `CumReturnResid(15:30) / EST_VOL_prev` |
| `ShortTermReversal` | `Close_adj(t−1) / Close_adj(t−2) − 1` |
| `Momentum21d` | `Close_adj(t−2) / Close_adj(t−23) − 1` (skips t−1) |
| `DollarVolTrend` | `mean(DollarVol over t−1..t−5) / MDV_63_prev` |

**Dollar volume:** `DollarVol = Volume × SharesAdjFactor × Close_adj`

**Text features (described but not implemented):** earnings call sentiment (Loughran-McDonald lexicon), 10-Q risk factor change score (TF-IDF cosine similarity).

## Models Implemented

Five models trained; selected by validation weighted R² (weights = `sqrt(MDV_63_prev)` on 2014 data):

| Model | Hyperparameters Tuned |
|---|---|
| **Ridge** | alpha ∈ [0.001–1000], 25 values |
| **Random Forest** | max_depth ∈ {5,8}, min_samples_leaf ∈ {10,20}; final 80 estimators |
| **XGBoost** | max_depth ∈ {3–6}, lr ∈ {0.03,0.05,0.1}, n_estimators ∈ {100,200} |
| **ElasticNet** | alpha ∈ 10^[−5..−0.5], l1_ratio ∈ {0.8,0.9,0.95,1.0} |
| **MLP** | hidden layers, alpha, lr_init; 36 configs, early stopping |

**Best model: XGBoost** — saved to `MTH9899-Project/saved_model/best_model.pkl` (requires `xgboost` package).

**Saved artifacts:**
- `saved_model/best_model.pkl` — fitted XGBoost model
- `saved_model/feature_cols.pkl` — ordered list of 13 feature names
- `saved_model/scaler.pkl` — StandardScaler fitted on train set
- `saved_model/fit_target_mode.pkl` — `'vol_scaled'` (target normalization mode)

Additional considerations: survivorship bias avoidance; StandardScaler fitted only on train, applied to val/test.

## Loading Data

```python
import pandas as pd, glob

daily = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('DailyData/dat.*.csv'))], ignore_index=True)
# Intraday is ~1.1GB — load by date range or per-file as needed
intraday = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data_intraday/*.csv'))], ignore_index=True)
```

**Note:** The notebook uses `hw2/dat.*.csv` as the daily data path. Adjust glob pattern to match your local layout.

## Implementation

The full pipeline is in `MTH9899-Project/project.ipynb` (~4k lines). Key sections:
1. Utility functions and setup
2. Intraday + daily data loading
3. Target variable construction
4. Feature engineering (intraday then daily, with EDA histograms)
5. Model training and hyperparameter grid search
6. Validation evaluation and model comparison
7. Feature importance (permutation MDA, bin plots)
8. White paper visualizations (hyperparameter table, prediction bin plots, 30-day rolling correlation, drift plot)

To load and apply the saved best model:
```python
import pickle
from sklearn.preprocessing import StandardScaler

with open('MTH9899-Project/saved_model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('MTH9899-Project/saved_model/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
with open('MTH9899-Project/saved_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = df[feature_cols]
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
```
