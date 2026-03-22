"""Shared constants and helper functions used across all modules."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# Rolling window for per-Id time-series z-scores
TS_WINDOW = 252
TS_MIN_PERIODS = 60


def zscore_time_series_per_id(
    df,
    col,
    *,
    id_col='Id',
    date_col='Date',
    window=None,
    min_periods=None,
):
    """Rolling z-score per security using only *past* observations (no look-ahead).

    Uses shift(1) before the rolling window so the z-score at date D only depends
    on observations from dates strictly before D.
    """
    window = TS_WINDOW if window is None else window
    min_periods = TS_MIN_PERIODS if min_periods is None else min_periods
    work = df[[id_col, date_col, col]].copy()
    work['_row'] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values([id_col, date_col], kind='mergesort')

    def _roll_z(s):
        prev = s.shift(1)
        m = prev.rolling(window, min_periods=min_periods).mean()
        sd = prev.rolling(window, min_periods=min_periods).std()
        return (s - m) / sd.replace(0, np.nan)

    work['_zts'] = work.groupby(id_col, sort=False)[col].transform(_roll_z)
    work = work.sort_values('_row')
    return pd.Series(work['_zts'].to_numpy(), index=df.index)


def winsorize_mad(df, col, n_mad=5):
    """Winsorize at ±n_mad * MAD cross-sectionally per date."""
    mad = df.groupby('Date')[col].transform(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    mad = mad.replace(0, np.nan)
    med = df.groupby('Date')[col].transform('median')
    lo = med - n_mad * mad
    hi = med + n_mad * mad
    return df[col].clip(lower=lo.fillna(-np.inf), upper=hi.fillna(np.inf))


def winsorize_percentile(df, col, low=0.01, high=0.99):
    """Winsorize at [low, high] percentiles cross-sectionally per date."""
    lo = df.groupby('Date')[col].transform(lambda x: x.quantile(low))
    hi = df.groupby('Date')[col].transform(lambda x: x.quantile(high))
    return df[col].clip(lower=lo, upper=hi)


def zscore_cross_sectional(df, col):
    """Z-score within each date (cross-sectional). Returns NaN when std is 0."""
    return df.groupby('Date')[col].transform(
        lambda x: (x - x.mean()) / (x.std() or np.nan)
    )


def weighted_r2(y_true, y_pred, weight):
    """Weighted R² as per spec: weights = sqrt(MDV_63) for evaluation."""
    return r2_score(y_true, y_pred, sample_weight=weight)


# ---------------------------------------------------------------------------
# Feature vetting: Information Coefficient (IC) analysis
# ---------------------------------------------------------------------------

def compute_feature_ic(df, feature_cols, target_col='Target_model', date_col='Date',
                       min_obs=30):
    """Per-date Spearman rank correlation of each feature against the target.

    IC (Information Coefficient) is the standard quantitative-finance metric
    for vetting individual predictors.  A feature with a high IC t-statistic
    demonstrates genuine predictive power, not just a spurious in-sample fit.

    Parameters
    ----------
    df : DataFrame containing features, target, and date columns.
    feature_cols : list of feature column names to evaluate.
    target_col : target column name (default 'Target_model').
    date_col : date column name (default 'Date').
    min_obs : int, minimum number of valid paired observations required to
        compute IC for a given date.  Dates with fewer observations yield NaN.
        Default 30 — below this threshold the Spearman ρ estimate is too noisy
        to be informative (standard rule-of-thumb for rank correlations).

    Returns
    -------
    ic_df : DataFrame (n_dates × n_features) of daily Spearman ICs.
    ic_summary : DataFrame with columns Mean_IC, Std_IC, IC_tstat, ICIR,
                 N_dates; sorted by |Mean_IC| descending.
    """
    records = []
    for date, grp in df.groupby(date_col):
        row = {date_col: date}
        valid_target = grp[target_col].notna()
        for feat in feature_cols:
            valid = valid_target & grp[feat].notna()
            if valid.sum() < min_obs:
                row[feat] = np.nan
            else:
                r, _ = stats.spearmanr(grp.loc[valid, feat], grp.loc[valid, target_col])
                row[feat] = r
        records.append(row)

    ic_df = pd.DataFrame(records).set_index(date_col)

    n = ic_df.notna().sum()
    mean_ic = ic_df.mean()
    std_ic = ic_df.std()
    ic_summary = pd.DataFrame({
        'Mean_IC': mean_ic,
        'Std_IC': std_ic,
        'IC_tstat': mean_ic / (std_ic / np.sqrt(n)),
        'ICIR': mean_ic / std_ic,
        'N_dates': n,
    }).sort_values('Mean_IC', key=abs, ascending=False)

    return ic_df, ic_summary


# ---------------------------------------------------------------------------
# Bias analysis: vol and size exposure of model predictions
# ---------------------------------------------------------------------------

def bucket_mean_prediction(df, preds, char_col, n_buckets=10):
    """Mean model prediction per equal-frequency bucket of a risk characteristic.

    Splits ``char_col`` into ``n_buckets`` quantile buckets (e.g. n_buckets=10
    gives deciles, n_buckets=5 gives quintiles) and returns the mean prediction
    in each bucket.  A flat result (all bucket means near zero) shows the model
    has no systematic tilt toward or against stocks ranked by that
    characteristic (e.g., volatility, liquidity).

    Parameters
    ----------
    df : DataFrame containing at least ``char_col``.
    preds : array-like of model predictions, aligned with ``df``.
    char_col : str, name of the risk-characteristic column to bucket by.
    n_buckets : int, number of equal-frequency buckets (default 10).

    Returns
    -------
    DataFrame with columns: bucket (0-based int), mean_pred, count.
    """
    tmp = df[[char_col]].copy()
    tmp['_pred'] = preds
    tmp['bucket'] = pd.qcut(tmp[char_col], q=n_buckets, labels=False, duplicates='drop')
    result = tmp.groupby('bucket')['_pred'].agg(['mean', 'count']).reset_index()
    result.columns = ['bucket', 'mean_pred', 'count']
    return result


def spearman_vs_characteristics(preds, char_dict):
    """Spearman rank correlation of predictions against risk characteristics.

    Parameters
    ----------
    preds : array-like of model predictions.
    char_dict : dict mapping characteristic label -> array-like of values.

    Returns a DataFrame with columns: characteristic, spearman_r, p_value.
    """
    rows = []
    for label, values in char_dict.items():
        mask = ~(np.isnan(preds) | np.isnan(values))
        r, p = stats.spearmanr(np.asarray(preds)[mask], np.asarray(values)[mask])
        rows.append({'Characteristic': label, 'Spearman_r': r, 'p_value': p})
    return pd.DataFrame(rows)
