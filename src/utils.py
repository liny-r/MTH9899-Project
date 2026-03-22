"""Shared constants and helper functions used across all modules."""

import numpy as np
import pandas as pd
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
