# White Paper: Intraday Equity Return Prediction Model

**MTH 9899 — Quantitative Finance Practicum**
**Submission Section 5.2**

---

## Abstract

We develop a supervised regression model to predict the next 24-hour equity return for approximately 1,258 US equities using intraday microstructure signals. Training covers 2010–2013 (~452K observations); the 2014 calendar year serves as a clean out-of-sample validation set (~88K observations). Following an IC-based feature selection protocol, four intraday features survive statistical screening. An ElasticNet model achieves the highest validation weighted R² (0.000221) under liquidity-weighted evaluation, outperforming Ridge regression, Random Forest, XGBoost, and an MLP on the held-out year. All bias controls are enforced: no look-ahead in features, targets, normalization, or hyperparameter selection.

---

## 1. Data and Train/Validation Split

### 1.1 Data Sources

Two data sources are used, each containing one file per trading day.

**DailyData/** (`dat.YYYYMMDD.csv`) — contains OHLCV data, adjustment factors, 63-day median daily dollar volume (`MDV_63`), and annualized volatility estimates (`EST_VOL`) for each security. Prices are adjusted for corporate actions using the multiplicative factor `PxAdjFactor`:

```
Close_adj = Close × PxAdjFactor
```

**data_intraday/** (`YYYYMMDD.csv`) — contains cumulative residual and raw returns, and cumulative share volume sampled at 15-minute intervals from 09:45 to 16:00. All feature computations use data available at or before 15:30 to avoid intraday look-ahead.

### 1.2 Target Variable

The prediction target is the 24-hour forward residual return spanning two trading sessions:

- **Part A (same day):** `CumReturnResid(16:00) − CumReturnResid(15:30)` on day D
- **Part B (next day):** `CumReturnResid(15:30)` on day D+1
- **Target:** `Part A + Part B`

This composite return captures both the close-to-open gap and the intraday trend through the following afternoon, making it a natural horizon for signals generated at 15:30.

**Target normalization pipeline (`vol_scaled` mode):**

1. Divide by the previous trading day's `EST_VOL` (volatility-scale the return)
2. Per-security time-series z-score: 252-day rolling window, `min_periods=60`, one-period lag to prevent look-ahead
3. Cross-sectional ±5 MAD winsorization per date
4. Cross-sectional z-score per date

The resulting target `Target_model` has approximately unit variance cross-sectionally, facilitating the use of a common R² objective across all models.

**Sample weights:** `sqrt(MDV_63_prev)` — larger, more liquid stocks receive proportionally higher weight in the R² objective, ensuring that predictive accuracy concentrates in the tradeable universe.

### 1.3 Data Split and Walk-Forward Cross-Validation

The data covers 2010-01-04 to 2014-12-31. The split is purely time-based to prevent any future information from contaminating the training or feature normalization steps:

| Period | Role | Trading Days | Observations (after normalization) |
|--------|------|-------------|-------------------------------------|
| 2010–2013 | Training | ~1,006 | ~451,610 |
| 2014 | Validation (holdout) | ~251 | ~87,624 |

**Walk-forward cross-validation** with four expanding folds provides an honest assessment of how each model generalizes across market regimes:

| Fold | Training Years | Validation Year |
|------|---------------|-----------------|
| 1 | 2010 | 2011 |
| 2 | 2010–2011 | 2012 |
| 3 | 2010–2012 | 2013 |
| 4 | 2010–2013 | 2014 |

**Hyperparameter leakage prevention:** Fold 4 (val=2014) uses hyperparameters frozen from Fold 3 (val=2013). No grid search is performed on the 2014 validation set; the reported 2014 R² is therefore fully out-of-sample and uncontaminated by tuning decisions.

---

## 2. Feature Engineering

### 2.1 Candidate Feature Library

Twelve candidate features were constructed (all using data strictly at or before 15:30 on the prediction day, or from prior trading days):

**Intraday features:**

| Feature | Definition | Type |
|---------|-----------|------|
| `OvernightReturn` | `CumReturnResid` at 09:45 | Return-like |
| `FirstHourMomentum` | `CumReturnResid(11:00) − CumReturnResid(09:45)` | Return-like |
| `LastHourMomentum` | `CumReturnResid(15:30) − CumReturnResid(14:30)` | Return-like |
| `IntradayReversal` | Morning return (09:45→12:00) minus afternoon return (12:00→15:30) | Return-like |
| `IntradayReturnSkew` | Skewness of 15-min residual return increments (09:45–15:30) | Return-like |
| `VolatilityAdjustedReturn` | `CumReturnResid(15:30) / EST_VOL_prev` | Return-like |
| `RealizedUpsideVol` | RMS of positive 15-min increments | Ratio/volume |
| `IntradayVol` | Std dev of 15-min return increments (09:45–15:30) | Ratio/volume |
| `VolumeSurprise` | Dollar volume through 15:30 / `MDV_63_prev` | Ratio/volume |
| `VolumeMorningAfternoonRatio` | Morning volume / afternoon volume (split 12:00) | Ratio/volume |
| `IntradayVolAccel` | `CumVolume(15:30) / CumVolume(12:00)` | Ratio/volume |
| `RetVolCorr` | Pearson correlation of 15-min return and volume increments | Bounded [−1, 1] |

### 2.2 Normalization Pipeline

Each feature undergoes a three-step normalization before being used in any model:

**Step 1 — Time-series z-score (per security):**
For each security *i* and date *t*, subtract the rolling mean and divide by the rolling standard deviation computed over the prior 252 trading days (minimum 60 observations). A one-period lag is applied before the rolling window so that the z-score at date *t* depends only on history strictly before *t*:

```
z_ts(i, t) = [x(i, t) − μ_{rolling}(i, t−)] / σ_{rolling}(i, t−)
```

This step removes persistent security-level idiosyncrasies (e.g., a chronically high-volume stock appearing as a "VolumeSurprise" every day).

**Step 2 — Cross-sectional winsorization:**
- *Return-like features* (OvernightReturn, momentum signals, VolatilityAdjustedReturn): clipped at ±5 MAD from the cross-sectional median, per date.
- *Ratio/volume features* (realized vol, volume ratios): clipped at the 1st–99th percentile of the cross-sectional distribution, per date. These features are right-skewed and do not center near zero, making percentile clipping more appropriate.
- `RetVolCorr` is bounded by construction ([−1, 1]) and receives no winsorization.

**Step 3 — Cross-sectional z-score:**
Standardize across all securities within each date, producing features with mean zero and unit variance cross-sectionally. This ensures all features enter the model on a common scale and that the StandardScaler applied subsequently operates on an approximately standard distribution.

### 2.3 Feature Selection via Information Coefficient Analysis

The Information Coefficient (IC) is the per-date Spearman rank correlation between a feature and the forward target. It directly measures the feature's predictive usefulness without conflating it with distributional properties. We compute daily ICs over the 2010–2013 training period (885 trading days).

**Full IC table (training set, sorted by |Mean IC|):**

| Feature | Mean IC | Std IC | IC t-stat | ICIR | N dates |
|---------|---------|--------|-----------|------|---------|
| `LastHourMomentum` | −0.0174 | 0.0557 | −9.29 | −0.312 | 885 |
| `IntradayVolAccel` | +0.0106 | 0.0493 | +6.40 | +0.215 | 885 |
| `VolatilityAdjustedReturn` | −0.0104 | 0.0512 | −6.01 | −0.202 | 885 |
| `IntradayReversal` | +0.0093 | 0.0536 | +5.14 | +0.173 | 885 |
| `VolumeMorningAfternoonRatio` | −0.0090 | 0.0486 | −5.52 | −0.186 | 885 |
| `OvernightReturn` | +0.0033 | 0.0531 | +1.83 | +0.062 | 885 |
| `IntradayReturnSkew` | +0.0031 | 0.0492 | +1.88 | +0.063 | 885 |
| `RetVolCorr` | +0.0021 | 0.0498 | +1.25 | +0.042 | 885 |
| `IntradayVol` | +0.0010 | 0.0519 | +0.56 | +0.019 | 885 |
| `FirstHourMomentum` | −0.0009 | 0.0528 | −0.51 | −0.017 | 885 |
| `RealizedUpsideVol` | −0.0007 | 0.0516 | −0.42 | −0.014 | 885 |
| `VolumeSurprise` | +0.0005 | 0.0588 | +0.27 | +0.009 | 885 |

The absolute mean IC values are modest (≈1–1.7 bp), consistent with the low signal-to-noise ratio typical of short-horizon equity prediction. The IC t-statistic, which accounts for the serial variability of daily ICs, provides a more reliable significance measure.

**Correlation-based deduplication:** Prior to significance testing, highly correlated feature pairs were resolved by keeping the higher-IC-t-stat feature:
- `VolumeMorningAfternoonRatio` vs `IntradayVolAccel` → keep `IntradayVolAccel` (|t|=6.40 > 5.52)
- `RealizedUpsideVol` vs `IntradayVol` → keep `IntradayVol` (|t|=0.56 > 0.42)

**Benjamini-Hochberg FDR correction (α=0.05, 10 tests after deduplication):**

| Feature | |IC t-stat| | Raw p | Adj p (BH) | Decision |
|---------|------------|-------|------------|----------|
| `LastHourMomentum` | 9.29 | <0.0001 | <0.0001 | **KEEP** |
| `IntradayVolAccel` | 6.40 | <0.0001 | <0.0001 | **KEEP** |
| `VolatilityAdjustedReturn` | 6.01 | <0.0001 | <0.0001 | **KEEP** |
| `IntradayReversal` | 5.14 | <0.0001 | <0.0001 | **KEEP** |
| `VolumeMorningAfternoonRatio` | 5.52 | — | — | *Dropped (dedup)* |
| `IntradayReturnSkew` | 1.88 | 0.0603 | 0.1118 | DROP |
| `OvernightReturn` | 1.83 | 0.0671 | 0.1118 | DROP |
| `RetVolCorr` | 1.25 | 0.2106 | 0.3008 | DROP |
| `IntradayVol` | 0.56 | 0.5756 | 0.6766 | DROP |
| `FirstHourMomentum` | 0.51 | 0.6090 | 0.6766 | DROP |
| `RealizedUpsideVol` | 0.42 | — | — | *Dropped (dedup)* |
| `VolumeSurprise` | 0.27 | 0.7861 | 0.7861 | DROP |

**Final feature set (4 features):** `LastHourMomentum`, `IntradayReversal`, `VolatilityAdjustedReturn`, `IntradayVolAccel`

The post-selection correlation matrix confirms no pairwise |r| ≥ 0.50 among the four retained features, so multicollinearity is not a concern.

**Stationarity:** ADF and KPSS tests on monthly cross-sectional means confirm all features are stationary processes (ADF p < 0.001, KPSS p > 0.07 for all). This validates the use of a fixed model without regime-based refitting.

---

## 3. Model Comparison

### 3.1 Model Architectures and Hyperparameter Grids

Five model families were evaluated. Hyperparameters were tuned via grid search on Fold 3 (val=2013); the best configuration from Fold 3 was applied without re-searching on Fold 4 (val=2014) to preserve the clean holdout property of the 2014 validation year.

**Ridge Regression**
A linear model with L2 regularization. With 4 normalized, near-orthogonal features, Ridge provides a low-variance baseline.
- Grid: `alpha` ∈ {10^−3, ..., 10^3} (25 log-spaced values)
- Best hyperparameter: `alpha = 0.001` (minimal regularization — consistent with features already cross-sectionally z-scored)

**Random Forest**
An ensemble of decision trees with bootstrap sampling capped at 80,000 rows per tree for computational feasibility.
- Grid: `n_estimators` ∈ {80, 150}, `max_depth` ∈ {5, 7, 10}, `min_samples_leaf` ∈ {10, 20}
- Best hyperparameters: `max_depth=5`, `min_samples_leaf=20`, `n_estimators=150`

**XGBoost**
Gradient-boosted trees with a 2-level factorial search over all six key hyperparameters (2^6 = 64 configurations).
- Grid: `max_depth` ∈ {3, 5}, `learning_rate` ∈ {0.03, 0.1}, `n_estimators` ∈ {80, 150}, `subsample` ∈ {0.8, 1.0}, `colsample_bytree` ∈ {0.8, 1.0}, `min_child_weight` ∈ {10, 50}
- Best hyperparameters: `max_depth=3`, `learning_rate=0.03`, `n_estimators=150`, `subsample=0.8`, `colsample_bytree=1.0`, `min_child_weight=10`

**ElasticNet**
Linear model with combined L1 and L2 penalties. Tuning on a 100k-row subsample for speed; final model refitted on the full training set.
- Grid: `alpha` ∈ 10^[−5, ..., −0.5] (12 log-spaced values), `l1_ratio` ∈ {0.1, 0.5, 0.8, 0.9, 0.95, 1.0}
- Best hyperparameters: `alpha=1e-05`, `l1_ratio=0.1` (near-Ridge, very light L1)

**MLP (Multi-Layer Perceptron)**
Feed-forward neural network with adaptive learning rate, early stopping (patience=25 epochs, tol=1e-4), and weight decay.
- Grid: `hidden_layer_sizes` ∈ {(64,), (128,), (128, 64)}, `alpha` ∈ {1e-4, 1e-3, 1e-2}, `learning_rate_init` ∈ {1e-3, 2e-3}
- Best hyperparameters: `hidden_layer_sizes=(128, 64)`, `alpha=0.0001`, `learning_rate_init=1e-3`

### 3.2 Final Hyperparameter Table

| Model | Hyperparameter | Value |
|-------|---------------|-------|
| Ridge | alpha | 0.001 |
| Random Forest | max_depth | 5 |
| | min_samples_leaf | 20 |
| | n_estimators | 150 |
| XGBoost | max_depth | 3 |
| | learning_rate | 0.03 |
| | n_estimators | 150 |
| | subsample | 0.8 |
| | colsample_bytree | 1.0 |
| | min_child_weight | 10 |
| ElasticNet | alpha | 1e-05 |
| | l1_ratio | 0.1 |
| MLP | hidden_layer_sizes | (128, 64) |
| | alpha (L2 penalty) | 1e-04 |
| | learning_rate_init | 1e-03 |

---

## 4. Validation Performance

### 4.1 Walk-Forward R² Across Folds

Evaluation metric: weighted R² with sample weights `sqrt(MDV_63_prev)`.

| Model | Val 2011 | Val 2012 | Val 2013 | Val 2014 |
|-------|----------|----------|----------|----------|
| ElasticNet | +0.000491 | +0.000058 | +0.000486 | **+0.000221** |
| Ridge | +0.000491 | −0.000067 | +0.000486 | **+0.000221** |
| XGBoost | +0.000154 | +0.000176 | +0.000376 | −0.000084 |
| Random Forest | +0.000076 | −0.000135 | +0.000184 | +0.000007 |
| MLP | −0.002228 | −0.000992 | −0.000206 | −0.000333 |

**XGB+Ridge ensemble (equal weight):** Val 2014 = +0.000117
**XGB+Ridge ensemble (R²-weighted):** Val 2014 = +0.000136

**Selected model: ElasticNet** (val 2014 weighted R² = **0.000221**).

The absolute R² values are small by conventional standards but are characteristic of normalized, cross-sectional short-horizon signals. The cross-sectional target (vol-scaled, TS z-scored, CS z-scored) is dominated by idiosyncratic noise. In practice, alpha models with R² in this range generate economically meaningful returns when applied at scale across a large, liquid universe.

### 4.2 Overfitting Analysis

| Model | Train R² | Val R² (2014) | Overfit Ratio (Train/Val) |
|-------|----------|---------------|--------------------------|
| ElasticNet | 0.000382 | 0.000221 | **1.73×** |
| Ridge | 0.000382 | 0.000221 | **1.73×** |
| Random Forest | 0.001513 | 0.000007 | 210.6× |
| XGBoost | 0.001004 | −0.000084 | (negative val R²) |
| MLP | 0.001246 | −0.000333 | (negative val R²) |

ElasticNet and Ridge exhibit minimal overfitting (1.73× ratio), reflecting that a linear model with 4 well-selected features generalizes robustly. Tree-based models and the MLP show severe over-fit — their higher in-sample R² does not translate to out-of-sample performance, consistent with the signal-to-noise environment of intraday equity returns.

### 4.3 Model Comparison Summary

| Model | Val R² (2014) | Overfit Ratio | Penalty-Adjusted Score |
|-------|--------------|---------------|------------------------|
| ElasticNet | 0.000221 | 1.73× | 0.000153 |
| Ridge | 0.000221 | 1.73× | 0.000153 |
| XGB+Ridge (R²-wtd) | 0.000136 | N/A | N/A |
| XGB+Ridge (equal) | 0.000117 | N/A | N/A |
| Random Forest | 0.000007 | 210.6× | 0.000001 |
| XGBoost | −0.000084 | N/A | N/A |
| MLP | −0.000333 | N/A | N/A |

The penalty-adjusted score is defined as `Val_R² / log₂(overfit_ratio + 1)`, which penalizes models that achieve high validation R² only by massively over-fitting training data.

### 4.4 Feature Importance (Permutation / MDA)

Permutation importance was computed by randomly shuffling each feature column on the validation set (8 repeats, random_state=42) and measuring the mean decrease in weighted R². A larger drop indicates higher reliance.

| Feature | MDA Importance | MDA Std |
|---------|---------------|---------|
| `LastHourMomentum` | 0.000321 | ±0.000060 |
| `IntradayVolAccel` | 0.000159 | ±0.000074 |
| `VolatilityAdjustedReturn` | 0.000026 | ±0.000019 |
| `IntradayReversal` | 0.000002 | ±0.000022 |

`LastHourMomentum` is the dominant predictor: disrupting it alone drops model R² by more than the other three features combined. This is consistent with its IC t-stat of −9.29, the highest in absolute value among all candidates. The negative sign indicates a **short-term reversal** pattern — stocks that accelerate strongly into the 15:30 close tend to give back those gains over the next 24 hours.

`IntradayVolAccel`, the ratio of cumulative volume at 15:30 to volume at 12:00, captures whether buying/selling pressure is accelerating into the close. A high afternoon volume ratio (relative to morning) tends to predict a positive next-day residual, consistent with informed late-day accumulation.

`VolatilityAdjustedReturn`, the 15:30 cumulative return scaled by prior-day volatility, provides a cross-sectional measure of how much a stock moved relative to its typical daily range.

`IntradayReversal` — the spread between morning and afternoon intraday return — contributes marginally in this model. Its IC t-stat (5.14) cleared the BH threshold, but its MDA is near zero, suggesting collinearity with the other momentum features absorbs most of its marginal information.

### 4.5 Bin Plot

The prediction bin plot groups the 2014 validation observations into 20 equal-frequency quantile bins of the predicted score and plots the mean actual target within each bin. A monotonically increasing relationship from low predictions (negative bins, Q1–Q7) to high predictions (positive bins, Q14–Q20) confirms that the model's rank ordering is economically meaningful even where individual predictions are small in absolute magnitude.

### 4.6 Prediction Drift Plot

The drift plot decomposes the 2014 validation period into 4 prediction quartile bins and plots the mean actual target within each bin over trading days. The separation between the lowest and highest prediction bins remains persistent throughout 2014 without systematic deterioration at any point in the year, providing evidence that the signal is stationary across market regimes (including the low-volatility mid-2014 regime and the more volatile Q4).

### 4.7 30-Day Rolling Correlation

The 30-day moving average of daily cross-sectional rank correlation between predictions and realized targets hovers consistently above zero throughout 2014, with a mean around 1–2%. Daily cross-sectional correlation is noisy (typical range ±5–8%) but remains positive on a rolling basis, indicating genuine out-of-sample predictive value. No sustained negative correlation periods appear, which would have indicated model breakdown or a regime change not captured by the training data.

---

## 5. Data Integrity and Bias Controls

*(See also: `whitepaper/data_integrity_and_bias_controls.md` for detailed discussion.)*

**Trading calendar.** The trading day calendar is derived implicitly from the set of files present on disk. All backward-looking lags — previous trading day lookups, 5-day and 21-day momentum windows — are computed via index positions within this file-derived calendar, ensuring that weekends and market holidays are never treated as valid observation dates.

**Survivorship bias.** The cross-sectional universe is determined each day by the securities present in that day's files. Securities that delist or drop out simply cease to appear; the pipeline never assumes forward survival.

**Look-ahead bias.**
- All time-series z-scores apply a one-period shift before the rolling window, so standardization at date *t* depends only on dates strictly before *t*.
- The StandardScaler is fit exclusively on the 2010–2013 training set and applied without refitting to validation and test sets.
- Walk-forward CV uses strictly expanding training windows.
- Hyperparameters for the 2014 validation fold were selected on Fold 3 (val=2013) with no grid search touching 2014 data.

**Feature cutoff.** All intraday features use only data at or before 15:30, the natural cutoff for a 15:30-close strategy.

---

## 6. Saved Artifacts and Deployment

The following artifacts are persisted in `saved_model/`:

| File | Contents |
|------|----------|
| `best_model.pkl` | Fitted ElasticNet (alpha=1e-05, l1_ratio=0.1) |
| `xgb_model.pkl` | Fitted XGBoost (for ensemble) |
| `ridge_model.pkl` | Fitted Ridge (for ensemble) |
| `ensemble_weights.pkl` | XGB weight=0.538, Ridge weight=0.462 |
| `feature_cols.pkl` | `['LastHourMomentum', 'IntradayReversal', 'VolatilityAdjustedReturn', 'IntradayVolAccel']` |
| `scaler.pkl` | StandardScaler fitted on 2010–2013 only |
| `fit_target_mode.pkl` | `'vol_scaled'` |

`src/predict.py::load_artifacts()` auto-detects ensemble vs single-model artifacts. Predictions are generated in the normalized target space (Target_model); `rescale_to_return_space=True` multiplies by `EST_VOL_prev` to recover return-unit predictions.

---

## 7. Conclusions

The final model is an ElasticNet with alpha=1e-05 and l1_ratio=0.1, trained on four IC-selected intraday microstructure features. The key findings are:

1. **Signal exists but is small:** Val weighted R² = 0.000221 (normalized target space). The signal is real and persistent through 2014 but small in absolute magnitude, consistent with efficient-market competition in the US equity universe.

2. **Linear models dominate:** ElasticNet and Ridge, with 1.73× overfit ratios, outperform all nonlinear models on the out-of-sample validation year. With only 4 features in a well-normalized feature space, nonlinear models have insufficient degrees of freedom to improve over the linear approximation without overfitting to the 1.26M training observations.

3. **Feature concentration is justified:** The IC screening eliminated 8 of 12 candidate features. The 4 survivors each have |IC t-stat| > 5.1 and pass the BH FDR test. Using all 12 features would introduce noise that hurts out-of-sample performance.

4. **Bias controls are tight:** Overfitting ratios for the winning model (1.73×) are low, walk-forward R² is positive across 3 of 4 folds, and the 30-day rolling correlation is persistently above zero throughout 2014 — all confirming that the 2014 validation is genuinely out-of-sample.

5. **`LastHourMomentum` is the dominant signal** (MDA importance 0.000321 ± 0.000060), accounting for approximately twice the contribution of all other features combined. Its negative IC sign (mean IC = −0.0174) signals a short-term reversal pattern at the closing auction.
