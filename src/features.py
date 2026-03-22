"""Feature engineering: intraday + daily features, normalization pipeline, and EDA helpers."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from .utils import (
    TS_WINDOW, TS_MIN_PERIODS,
    zscore_time_series_per_id, winsorize_mad, winsorize_percentile, zscore_cross_sectional,
)
from .data import load_day_pivoted, build_daily_prev, INTRADAY_DIR

# Key intraday timestamps
TIME_0945 = '09:45'
TIME_1100 = '11:00'
TIME_1200 = '12:00'
TIME_1430 = '14:30'
TIME_1530 = '15:30'

# Features using ±5 MAD winsorization (return-like, symmetric)
MAD_FEATURES = [
    'OvernightReturn', 'FirstHourMomentum', 'LastHourMomentum', 'IntradayReversal',
    'IntradayReturnSkew', 'VolatilityAdjustedReturn',
    'ShortTermReversal', 'Momentum21d', 'Momentum5d', 'DollarVolTrend',
]
# Features using 1st–99th percentile winsorization (ratio/volume, right-skewed)
PCT_FEATURES = [
    'RealizedUpsideVol', 'IntradayVol',
    'VolumeSurprise', 'VolumeMorningAfternoonRatio', 'IntradayVolAccel',
]
# Features with no winsorization (bounded by construction)
NO_WINSOR_FEATURES = ['RetVolCorr']

ALL_FEATURE_NAMES = MAD_FEATURES + PCT_FEATURES + NO_WINSOR_FEATURES


def features_single_day(r, v, date_int):
    """Compute raw intraday features from pivoted CumReturnResid / CumVolume DataFrames.

    Args:
        r: DataFrame of CumReturnResid, columns=time strings, index=Id
        v: DataFrame of CumVolume, columns=time strings, index=Id
        date_int: integer date (YYYYMMDD)

    Returns DataFrame with columns: Date, Id, raw feature columns,
        CumReturnResid_1530, CumVolume_* (for later use in vol-adjusted features).
    """
    out = pd.DataFrame(index=r.index)
    out['Date'] = date_int
    out['Id'] = out.index

    # 1. Overnight return (gap/overnight sentiment)
    if TIME_0945 in r.columns:
        out['OvernightReturn'] = r[TIME_0945].values

    # 2. First-hour momentum (09:45 → 11:00; fall back to 09:45 → 12:00)
    if TIME_0945 in r.columns:
        if TIME_1100 in r.columns:
            out['FirstHourMomentum'] = (r[TIME_1100] - r[TIME_0945]).values
        elif TIME_1200 in r.columns:
            out['FirstHourMomentum'] = (r[TIME_1200] - r[TIME_0945]).values

    # 3. Last-hour momentum (14:30 → 15:30)
    if TIME_1430 in r.columns and TIME_1530 in r.columns:
        out['LastHourMomentum'] = (r[TIME_1530] - r[TIME_1430]).values

    # CumReturnResid at 15:30 (used to compute VolatilityAdjustedReturn after daily merge)
    if TIME_1530 in r.columns:
        out['CumReturnResid_1530'] = r[TIME_1530].values

    # 4. Intraday reversal (morning return − afternoon return)
    if all(t in r.columns for t in [TIME_0945, TIME_1200, TIME_1530]):
        morning = r[TIME_1200] - r[TIME_0945]
        afternoon = r[TIME_1530] - r[TIME_1200]
        out['IntradayReversal'] = (morning - afternoon).values

    # 5–7. 15-min increments 09:45→15:30: skew, upside realized vol, return-vol correlation
    times_ordered = sorted(
        [c for c in r.columns if TIME_0945 <= c <= TIME_1530],
        key=lambda t: (int(t[:2]), int(t[3:5])),
    )
    ret_inc = None
    if len(times_ordered) >= 2:
        ret_inc = r[times_ordered].diff(axis=1).iloc[:, 1:]
        out['IntradayReturnSkew'] = ret_inc.skew(axis=1).values
        arr = ret_inc.to_numpy(dtype=float)
        up_sq = np.where(arr > 0, arr * arr, np.nan)
        with np.errstate(all='ignore'):
            out['RealizedUpsideVol'] = np.sqrt(np.nanmean(up_sq, axis=1))
        # Symmetric intraday volatility (std of all 15-min increments, not just upside)
        out['IntradayVol'] = ret_inc.std(axis=1).values

    # Volume snapshot columns (MDV_63 scaling applied later after daily merge)
    if v is not None:
        v_a = v.reindex(r.index)
        for t, col in [(TIME_0945, 'CumVolume_0945'), (TIME_1200, 'CumVolume_1200'),
                       (TIME_1530, 'CumVolume_1530')]:
            if t in v.columns:
                out[col] = v_a[t].values

    # Return–volume correlation (corr of 15-min return increments with volume increments)
    if ret_inc is not None and v is not None:
        v_aligned = v.reindex(r.index)[times_ordered]
        vol_inc = v_aligned.diff(axis=1).iloc[:, 1:]
        out['RetVolCorr'] = ret_inc.corrwith(vol_inc, axis=1).values

    return out.reset_index(drop=True)


def _intraday_one_day(date_str, dev_dates, intraday_dir):
    date_int = int(date_str)
    if date_int not in dev_dates:
        return None
    r, v = load_day_pivoted(date_str, intraday_dir)
    if r is None:
        return None
    return features_single_day(r, v, date_int)


def build_intraday_features(date_strings, dev_dates, intraday_dir=INTRADAY_DIR):
    """Loop over all dev dates and compute raw intraday features.

    Args:
        date_strings: sorted list of all date strings
        dev_dates:    set/list of date ints to include
        intraday_dir: path to intraday CSV directory

    Returns concatenated DataFrame of raw intraday features.
    """
    dev_dates = set(dev_dates)
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_intraday_one_day)(date_str, dev_dates, intraday_dir)
        for date_str in tqdm(date_strings, desc='Intraday features')
    )
    dfs = [r for r in results if r is not None]
    return pd.concat(dfs, ignore_index=True)


def _daily_one_date(D, daily_by_date, prev_date, date_to_idx, date_list):
    if D not in prev_date:
        return None
    D_prev = prev_date[D]
    D_prev2 = prev_date.get(D_prev)
    if D_prev2 is None:
        return None
    idx = date_to_idx.get(D)
    if idx is None or idx < 23:
        return None
    D_prev23 = date_list[idx - 23]
    # Momentum5d: Close(t-2) / Close(t-7) - 1; t-7 is 5 days before t-2
    D_prev7  = date_list[idx - 7] if idx >= 7 else None

    if D_prev not in daily_by_date or D_prev2 not in daily_by_date or D_prev23 not in daily_by_date:
        return None

    d1  = daily_by_date[D_prev][['Close_adj', 'DollarVol', 'MDV_63']].rename(
        columns={'Close_adj': 'c1', 'DollarVol': 'dv1', 'MDV_63': 'MDV_63_1'})
    d2  = daily_by_date[D_prev2][['Close_adj']].rename(columns={'Close_adj': 'c2'})
    d23 = daily_by_date[D_prev23][['Close_adj']].rename(columns={'Close_adj': 'c23'})

    m = d1.join(d2, how='inner').join(d23, how='inner').reset_index()
    m['ShortTermReversal'] = (m['c1'] / m['c2']) - 1
    m['Momentum21d']       = (m['c2'] / m['c23']) - 1

    if D_prev7 is not None and D_prev7 in daily_by_date:
        d7 = daily_by_date[D_prev7][['Close_adj']].rename(columns={'Close_adj': 'c7'})
        m = m.join(d7, on='Id', how='left')
        m['Momentum5d'] = (m['c2'] / m['c7'].replace(0, np.nan)) - 1
    else:
        m['Momentum5d'] = np.nan

    dv_cols = {}
    d_ = D_prev
    for i in range(5):
        if d_ is None or d_ not in daily_by_date:
            break
        dv_cols[f'v{i}'] = daily_by_date[d_]['DollarVol']
        d_ = prev_date.get(d_)

    if len(dv_cols) == 5:
        dv_5d = pd.concat(dv_cols, axis=1)
        m = m.join(dv_5d.mean(axis=1).rename('avg_dv_5d'), on='Id', how='left')
        m['DollarVolTrend'] = m['avg_dv_5d'] / m['MDV_63_1'].replace(0, np.nan)
    else:
        m['DollarVolTrend'] = np.nan

    m = m[['Id', 'ShortTermReversal', 'Momentum21d', 'Momentum5d', 'DollarVolTrend']].copy()
    m['Date'] = D
    return m


def build_daily_features(feat_df, daily_all, date_list, prev_date):
    """Compute daily-only features requiring multiple lags of Close_adj / DollarVol.

    Features computed (all use data from dates strictly before D):
      - ShortTermReversal: Close_adj(t-1) / Close_adj(t-2) - 1
      - Momentum21d:       Close_adj(t-2) / Close_adj(t-23) - 1  (skips t-1)
      - Momentum5d:        Close_adj(t-2) / Close_adj(t-7)  - 1  (skips t-1)
      - DollarVolTrend:    mean(DollarVol, t-1..t-5) / MDV_63(t-1)

    Returns feat_df with these three columns merged in.
    """
    # Pre-index daily data by date to avoid O(N) scans per lookup
    daily_by_date = {
        d: grp.set_index('Id')[['Close_adj', 'DollarVol', 'MDV_63']]
        for d, grp in daily_all.groupby('Date')
    }
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    unique_dates = feat_df['Date'].unique()

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_daily_one_date)(D, daily_by_date, prev_date, date_to_idx, date_list)
        for D in tqdm(unique_dates, desc='Daily features')
    )
    rows = [r for r in results if r is not None]

    if rows:
        daily_feat_df = pd.concat(rows, ignore_index=True)
        feat_df = feat_df.merge(daily_feat_df, on=['Date', 'Id'], how='left')
    return feat_df


def attach_daily_prev(feat_df, daily_all, prev_date):
    """Merge previous-day EST_VOL and MDV_63 into feat_df, then compute
    volume-based and volatility-adjusted features that depend on them.

    Adds / overwrites: EST_VOL_prev, MDV_63_prev, VolumeSurprise,
    VolumeMorningAfternoonRatio, VolatilityAdjustedReturn.
    """
    # Drop stale columns to avoid duplicates on re-run
    for col in ['EST_VOL_prev', 'MDV_63_prev']:
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])

    daily_prev = build_daily_prev(daily_all, feat_df['Date'].unique(), prev_date)
    feat_df = feat_df.merge(daily_prev, on=['Date', 'Id'], how='left')

    feat_df['VolumeSurprise'] = (
        feat_df['CumVolume_1530'] / feat_df['MDV_63_prev'].replace(0, np.nan)
    )
    if 'CumVolume_0945' in feat_df.columns and 'CumVolume_1200' in feat_df.columns:
        _vm = feat_df['CumVolume_1200'] - feat_df['CumVolume_0945']
        _va = feat_df['CumVolume_1530'] - feat_df['CumVolume_1200']
        feat_df['VolumeMorningAfternoonRatio'] = _vm / _va.replace(0, np.nan)

    feat_df['VolatilityAdjustedReturn'] = (
        feat_df['CumReturnResid_1530'] / feat_df['EST_VOL_prev'].replace(0, np.nan)
    )
    # Afternoon-to-morning volume acceleration (ratio of cum volume at 15:30 vs 12:00)
    if 'CumVolume_1200' in feat_df.columns and 'CumVolume_1530' in feat_df.columns:
        feat_df['IntradayVolAccel'] = (
            feat_df['CumVolume_1530'] / feat_df['CumVolume_1200'].replace(0, np.nan)
        )
    return feat_df


def _normalize_one_feature(base_df, col):
    """Run the three-step normalization for a single feature column.

    Returns a Series of normalized values aligned to base_df's index.
    """
    tmp = base_df[['Date', 'Id', col]].copy()
    tmp['_ts'] = zscore_time_series_per_id(tmp, col)
    if col in PCT_FEATURES:
        tmp['_w'] = winsorize_percentile(tmp, '_ts')
    elif col in NO_WINSOR_FEATURES:
        tmp['_w'] = tmp['_ts']
    else:
        tmp['_w'] = winsorize_mad(tmp, '_ts', n_mad=5)
    return zscore_cross_sectional(tmp, '_w')


def normalize_features(feat_df):
    """Apply the three-step normalization pipeline to all features.

    Pipeline per feature:
      1. TS z-score per Id (rolling past-only window=252, min_periods=60)
      2. Cross-sectional ±5 MAD winsorization per date
      3. Cross-sectional z-score per date

    Returns (normalized_feat_df, feature_cols) where normalized_feat_df has
    only [Date, Id] + feature columns (intermediate columns dropped).
    """
    raw_cols = [c for c in ALL_FEATURE_NAMES if c in feat_df.columns]

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_normalize_one_feature)(feat_df, col)
        for col in tqdm(raw_cols, desc='Normalizing features')
    )

    out = feat_df[['Date', 'Id']].copy()
    for col, normed in zip(raw_cols, results):
        out[col] = normed

    feature_cols = list(raw_cols)
    out = out.dropna(subset=feature_cols)
    return out, feature_cols


# ---------------------------------------------------------------------------
# EDA helpers (optional; called from notebook for diagnostics)
# ---------------------------------------------------------------------------

def eda_stats_one_day(s, n_mad=5, min_n=30):
    """Cross-section on one date: shape, fraction outside ±n_mad·MAD."""
    v = pd.to_numeric(s, errors='coerce').dropna().to_numpy()
    n = len(v)
    empty = pd.Series({
        'count': n, 'skew': np.nan, 'excess_kurt': np.nan,
        'pct_clip_low': np.nan, 'pct_clip_high': np.nan, 'pct_clip_mad': np.nan,
        'mad': np.nan, 'width_pct_over_mad': np.nan,
    })
    if n < min_n:
        return empty
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    skew = float(pd.Series(v).skew())
    exk = float(pd.Series(v).kurtosis())
    if not np.isfinite(mad) or mad == 0:
        return pd.Series({
            'count': n, 'skew': skew, 'excess_kurt': exk,
            'pct_clip_low': np.nan, 'pct_clip_high': np.nan, 'pct_clip_mad': np.nan,
            'mad': mad if np.isfinite(mad) else np.nan, 'width_pct_over_mad': np.nan,
        })
    lo, hi = med - n_mad * mad, med + n_mad * mad
    fl = float((v < lo).mean())
    fh = float((v > hi).mean())
    p1, p99 = np.percentile(v, [1, 99])
    return pd.Series({
        'count': n, 'skew': skew, 'excess_kurt': exk,
        'pct_clip_low': fl, 'pct_clip_high': fh, 'pct_clip_mad': fl + fh,
        'mad': mad, 'width_pct_over_mad': float((p99 - p1) / (2 * n_mad * mad)),
    })


def run_feature_eda(feat_df, raw_cols=None):
    """Compute per-feature cross-sectional EDA summary across all dates.

    Returns a DataFrame indexed by feature name with median stats.
    """
    if raw_cols is None:
        raw_cols = [c for c in ALL_FEATURE_NAMES if c in feat_df.columns]

    rows = []
    for col in raw_cols:
        if col not in feat_df.columns:
            continue
        day_stats = pd.DataFrame([
            eda_stats_one_day(ser)
            for _, ser in feat_df.groupby('Date', sort=False)[col]
        ])
        rows.append({
            'feature': col,
            'days': len(day_stats),
            'median_n': day_stats['count'].median(),
            'median_skew': day_stats['skew'].median(),
            'median_excess_kurt': day_stats['excess_kurt'].median(),
            'median_pct_clip_5MAD': day_stats['pct_clip_mad'].median(),
            'median_asymmetry': (
                day_stats['pct_clip_high'] - day_stats['pct_clip_low']
            ).abs().median(),
            'median_width_pct_over_mad': day_stats['width_pct_over_mad'].median(),
            'pct_days_mad_zero': (
                day_stats['mad'].isna() | (day_stats['mad'] == 0)
            ).mean() * 100,
        })
    return pd.DataFrame(rows).set_index('feature')
