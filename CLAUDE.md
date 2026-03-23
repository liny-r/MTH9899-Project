# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project runs in a Docker DevContainer. The container provides Python 3 with: numpy, pandas, jupyter, statsmodels, scipy, scikit-learn, plotly, joblib, tqdm, python-dotenv, pytest.

- Start Jupyter: `jupyter notebook` or `jupyter lab`
- Run tests: `pytest`
- **Use `python3`, not `python`** — the `python` command is not available in this container.

## Project Status

**This project is complete and submitted.** All deliverables for sections 5.1, 5.2, and 5.3 of `project_assignment.pdf` are finished.

### 5.1 — Code and Model (`main.py` + `saved_model/`)

`main.py` is the CLI entry point with two modes:

- **Mode 1 — Feature generation:** reads raw daily + intraday data, constructs and normalizes all 16 features, writes one CSV per trading date.
- **Mode 2 — Prediction:** reads feature CSVs from Mode 1, applies the saved model, writes prediction CSVs (`Date, Time, Id, Pred`).

```bash
# Mode 1: generate features from OOS data
python3 main.py -m 1 -i oos_data -o /tmp/features -s 20150101 -e 20151231

# Mode 2: generate predictions
python3 main.py -m 2 -i /tmp/features -o /tmp/preds -p saved_model -s 20150101 -e 20151231
```

Mode 1 auto-detects both naming conventions (`daily_data/`+`intraday_data/` for the holdout set, `DailyData/`+`data_intraday/` for training data). Predictions are output in normalized `Target_model` space as required by the grader evaluation.

**Saved model artifacts (`saved_model/`):**
| File | Contents |
|---|---|
| `best_model.pkl` | Fitted XGBoost model (best by validation weighted R²) |
| `feature_cols.pkl` | Ordered list of 16 feature names |
| `scaler.pkl` | StandardScaler fitted on 2010–2013 train set only |
| `fit_target_mode.pkl` | `'vol_scaled'` — target normalization mode |

### 5.2 — White Paper (`whitepaper/white_paper.pdf`)

The written report is at `whitepaper/white_paper.pdf` (source: `whitepaper/white_paper.md`). It covers data split methodology, all 16 features with IC analysis, normalization pipeline, model comparison (Ridge, RF, XGBoost, ElasticNet, MLP), hyperparameter choices, and validation performance metrics.

### 5.3 — Visualizations (Appendix of white paper)

All required visualizations (R² bin plot, drift plot, 30-day rolling correlation, MDA feature importance, IC stability) are embedded in the appendix of `whitepaper/white_paper.pdf`. The underlying figures are also available as standalone PNGs in `whitepaper/figures/`.

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
- Located at `oos_data/` (repo root). Contains `daily_data/` and `intraday_data/` subdirectories (same raw fields as training data, different folder names).
- Evaluated via `main.py` — Mode 1 auto-detects `daily_data/`+`intraday_data/` naming.

## Feature Engineering

**16 implemented features** (saved in `saved_model/feature_cols.pkl`). Normalization pipeline per feature: TS z-score (per-Id, 252-day rolling, past-only) → ±5 MAD cross-sectional winsorization → cross-sectional z-score. Volume/ratio features use 1st–99th percentile clipping instead of ±5 MAD.

**Intraday features (from data_intraday/, all at or before 15:30):**
| Feature | Definition |
|---|---|
| `OvernightReturn` | `CumReturnResid` at 09:45 |
| `FirstHourMomentum` | `CumReturnResid(12:00) − CumReturnResid(09:45)` |
| `LastHourMomentum` | `CumReturnResid(15:30) − CumReturnResid(14:30)` |
| `IntradayReversal` | Morning return − afternoon return (09:45→12:00 minus 12:00→15:30) |
| `IntradayReturnSkew` | Skewness of 15-min residual return increments (09:45–15:30) |
| `RealizedUpsideVol` | RMS of positive 15-min increments |
| `IntradayVol` | Std dev of 15-min return increments (09:45–15:30) |
| `IntradayVolAccel` | Ratio of afternoon vol to morning vol (split at 12:00) |
| `VolumeSurprise` | `(CumVolume(15:30) × SharesAdjFactor_prev × Close_adj_prev) / MDV_63_prev` — dollar volume normalized by MDV_63 |
| `VolumeMorningAfternoonRatio` | Morning vol / afternoon vol (split at 12:00) |
| `RetVolCorr` | Pearson corr of 15-min return and volume increments |

**Daily features (using previous trading day's data — no look-ahead):**
| Feature | Definition |
|---|---|
| `VolatilityAdjustedReturn` | `CumReturnResid(15:30) / EST_VOL_prev` |
| `ShortTermReversal` | `Close_adj(t−1) / Close_adj(t−2) − 1` |
| `Momentum21d` | `Close_adj(t−2) / Close_adj(t−23) − 1` (skips t−1) |
| `Momentum5d` | `Close_adj(t−2) / Close_adj(t−7) − 1` (skips t−1) |
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

**Best model: XGBoost** — saved to `saved_model/best_model.pkl` (requires `xgboost` package). An XGBoost+Ridge ensemble is also saved.

**Saved artifacts:**
- `saved_model/best_model.pkl` — fitted XGBoost (single best model)
- `saved_model/xgb_model.pkl` + `saved_model/ridge_model.pkl` + `saved_model/ensemble_weights.pkl` — ensemble
- `saved_model/feature_cols.pkl` — ordered list of 16 feature names
- `saved_model/scaler.pkl` — StandardScaler fitted on 2010–2013 train set only
- `saved_model/fit_target_mode.pkl` — `'vol_scaled'`

`src/predict.py::load_artifacts()` auto-detects ensemble vs single model. StandardScaler fitted only on train, applied to val/test.

## Loading Data

```python
import pandas as pd, glob

daily = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('DailyData/dat.*.csv'))], ignore_index=True)
# Intraday is ~1.1GB — load by date range or per-file as needed
intraday = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data_intraday/*.csv'))], ignore_index=True)
```

Use `src/data.py::load_daily_data()` and `build_date_maps()` instead of raw globs — they handle path construction and sorting consistently.

## Implementation

### Source modules (`src/`)
| Module | Purpose |
|---|---|
| `data.py` | Load daily/intraday CSVs, build date maps, build prev-day lookup |
| `features.py` | Build and normalize all 16 features |
| `target.py` | Construct and normalize the target variable |
| `train.py` | Model training, walk-forward CV, ensemble, artifact saving |
| `predict.py` | Load saved artifacts, generate predictions |
| `utils.py` | Shared helpers: z-score, winsorize, IC analysis, bias analysis |
| `visualization.py` | All plotting functions (IC stability, bin plots, drift, rolling corr, bias) |

### Entry points
- **`run.ipynb`** — full interactive pipeline: data load → features → model training → evaluation → white paper plots
- **`main.py`** — CLI for submission: Mode 1 (feature CSVs), Mode 2 (prediction CSVs)

### White paper drafts
Pre-written sections and content for the 5.2 white paper live in `whitepaper/`. Drop new `.md` files there as sections are drafted.

| File | Section covered |
|---|---|
| `data_integrity_and_bias_controls.md` | Trading calendar handling, survivorship bias, look-ahead bias |

### To apply the saved model to new data
```python
from src.predict import predict

preds = predict(df, model_dir='saved_model')  # df must contain the 16 feature columns
# Use rescale_to_return_space=True to convert predictions back to return units
```

### To run the CLI
```bash
# Mode 1: generate feature files from OOS data
python main.py -m 1 -i oos_data -o /tmp/features -s 20150101 -e 20151231

# Mode 2: generate predictions
python main.py -m 2 -i /tmp/features -o /tmp/preds -p saved_model -s 20150101 -e 20151231
```
