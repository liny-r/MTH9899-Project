# MTH9899 Final Project — Next-Day Residual Return Prediction

A machine learning pipeline to predict next-day residual stock returns at 15:30, trained on ~1,258 US equities over 2010–2014.

**Best model:** XGBoost (selected by liquidity-weighted R² on 2014 validation set)

---

## Deliverables

| Requirement | Location |
|---|---|
| **5.1 — Code and model** | `main.py` (CLI), `src/` (pipeline modules), `saved_model/` (artifacts) |
| **5.2 — White paper** | `whitepaper/white_paper.pdf` |
| **5.3 — Visualizations** | Appendix of `whitepaper/white_paper.pdf`; also `whitepaper/figures/` |

---

## Running the Model (Grader Instructions)

The CLI entry point is `main.py`. Use **Mode 1** to generate features, then **Mode 2** to generate predictions.

```bash
# Mode 1: Build features from raw OOS data
python3 main.py -m 1 -i oos_data -o /tmp/features -s 20150101 -e 20151231

# Mode 2: Generate predictions from features
python3 main.py -m 2 -i /tmp/features -o /tmp/preds -p saved_model -s 20150101 -e 20151231
```

**Mode 1** input (`-i`) must be the parent directory containing `daily_data/` and `intraday_data/` subdirectories (the holdout format). It also accepts the training-data naming convention (`DailyData/` + `data_intraday/`).

**Mode 2** output CSVs have columns: `Date, Time, Id, Pred`. Predictions are in normalized `Target_model` space for R² evaluation.

---

## Saved Model Artifacts (`saved_model/`)

| File | Contents |
|---|---|
| `best_model.pkl` | Fitted XGBoost model |
| `feature_cols.pkl` | Ordered list of 16 feature names |
| `scaler.pkl` | StandardScaler fitted on 2010–2013 train set only |
| `fit_target_mode.pkl` | `'vol_scaled'` — target normalization mode |

To apply the model programmatically:

```python
from src.predict import predict

# df must contain the 16 feature columns (already normalized)
preds = predict(df, model_dir='saved_model')

# Use rescale_to_return_space=True to convert predictions back to return units
preds_returns = predict(df, model_dir='saved_model', rescale_to_return_space=True)
```

---

## Repository Structure

```
MTH9899-Project/
├── main.py                     # CLI entry point (Mode 1: features, Mode 2: predictions)
├── src/
│   ├── data.py                 # Data loading and date map utilities
│   ├── target.py               # Target variable construction pipeline
│   ├── features.py             # Feature engineering (16 features: intraday + daily)
│   ├── train.py                # Model training, walk-forward CV, ensemble, artifact saving
│   ├── predict.py              # Inference with saved artifacts
│   ├── utils.py                # Shared utilities (weighted R², z-score, winsorize, IC)
│   └── visualization.py        # All plotting functions (IC, bin plots, drift, rolling corr)
├── tests/
│   ├── test_features.py
│   └── test_mode1_completeness.py
├── saved_model/
│   ├── best_model.pkl          # Fitted XGBoost model
│   ├── feature_cols.pkl        # Ordered list of 16 feature names
│   ├── scaler.pkl              # StandardScaler (train-fit only)
│   └── fit_target_mode.pkl     # 'vol_scaled'
├── whitepaper/
│   ├── white_paper.pdf         # Final written report (sections 5.2 + 5.3)
│   ├── white_paper.md          # Source markdown
│   └── figures/                # Standalone PNG exports of all plots
├── run.ipynb                   # Full interactive pipeline notebook
└── oos_data/                   # Held-out test data (2015)
    ├── daily_data/
    └── intraday_data/
```

Data directories (gitignored):
```
DailyData/        # dat.YYYYMMDD.csv — OHLCV + adjustment factors (training)
data_intraday/    # YYYYMMDD.csv — 15-min residual return snapshots (training)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Required packages: `numpy pandas scikit-learn xgboost joblib scipy statsmodels matplotlib plotly tqdm python-dotenv pytest`

Run tests:

```bash
pytest
```

---

## Prediction Task

**At 15:30 each trading day**, predict the 24-hour ahead residual return for each stock:
- **Part A:** Same-day 15:30→16:00 residual return
- **Part B:** Next trading day open→15:30 residual return
- `target = Part A + Part B`

**Train:** 2010–2013 (~1.26M rows) | **Validation:** 2014 (~315K rows) | **Test:** 2015 (held out)

**Sample weights:** `sqrt(MDV_63_prev)` for liquidity-weighted R² evaluation.

---

## Features (16 total)

All features use only data available at or before 15:30. Normalization pipeline: per-Id time-series z-score (252-day rolling, past-only) → cross-sectional ±5 MAD winsorization → cross-sectional z-score.

### Intraday Features (from `data_intraday/`)

| Feature | Definition |
|---|---|
| `OvernightReturn` | `CumReturnResid` at 09:45 |
| `FirstHourMomentum` | `CumReturnResid(12:00) − CumReturnResid(09:45)` |
| `LastHourMomentum` | `CumReturnResid(15:30) − CumReturnResid(14:30)` |
| `IntradayReversal` | Morning return − afternoon return (09:45→12:00 minus 12:00→15:30) |
| `IntradayReturnSkew` | Skewness of 15-min residual return increments |
| `RealizedUpsideVol` | RMS of positive 15-min increments |
| `IntradayVol` | Std dev of 15-min return increments |
| `IntradayVolAccel` | Ratio of afternoon vol to morning vol (split at 12:00) |
| `VolumeSurprise` | `(CumVolume(15:30) × SharesAdjFactor_prev × Close_adj_prev) / MDV_63_prev` |
| `VolumeMorningAfternoonRatio` | Morning volume / afternoon volume (split at 12:00) |
| `RetVolCorr` | Pearson correlation of 15-min return and volume increments |

### Daily Features (previous trading day — no look-ahead)

| Feature | Definition |
|---|---|
| `VolatilityAdjustedReturn` | `CumReturnResid(15:30) / EST_VOL_prev` |
| `ShortTermReversal` | `Close_adj(t−1) / Close_adj(t−2) − 1` |
| `Momentum5d` | `Close_adj(t−2) / Close_adj(t−7) − 1` (skips t−1) |
| `Momentum21d` | `Close_adj(t−2) / Close_adj(t−23) − 1` (skips t−1) |
| `DollarVolTrend` | `mean(DollarVol t−1..t−5) / MDV_63_prev` |

---

## Models

Five models trained; best selected by validation liquidity-weighted R²:

| Model | Hyperparameter Search |
|---|---|
| **XGBoost** ✓ | max_depth ∈ {3–6}, lr ∈ {0.03,0.05,0.1}, n_estimators ∈ {100,200} |
| Ridge | alpha ∈ logspace(−3, 3, 25) |
| Random Forest | max_depth ∈ {5,8}, min_samples_leaf ∈ {10,20}; 80 estimators |
| ElasticNet | alpha ∈ 10^[−5..−0.5], l1_ratio ∈ {0.8,0.9,0.95,1.0} |
| MLP | hidden layers, alpha, lr_init; 36 configs, early stopping |
