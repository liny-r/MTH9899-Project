"""Target variable construction and normalization pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from .utils import zscore_time_series_per_id, winsorize_mad, zscore_cross_sectional
from .data import build_daily_prev, INTRADAY_DIR


def build_target_single_day(date_str, next_date_str, intraday_dir=INTRADAY_DIR):
    """Build target for one date D: 24-hour return from 15:30 on D.

    Target = (CumReturnResid at 16:00 - CumReturnResid at 15:30) on D   [15:30→16:00 today]
           + CumReturnResid at 15:30 on next_date                        [16:00 D → 15:30 D+1]

    Returns DataFrame with columns: Date, Id, Target, part_a, part_b.
    Returns None if either file is missing.
    """
    path_today = Path(intraday_dir) / f'{date_str}.csv'
    path_next = Path(intraday_dir) / f'{next_date_str}.csv'
    if not path_today.exists() or not path_next.exists():
        return None

    full = pd.read_csv(path_today)
    full_next = pd.read_csv(path_next)

    r1530_t = (
        full[full['Time'].str.startswith('15:30')]
        .set_index('Id')['CumReturnResid'].rename('r1530')
    )
    r1600_t = (
        full[full['Time'].str.startswith('16:00')]
        .set_index('Id')['CumReturnResid'].rename('r1600')
    )
    r1530_next = (
        full_next[full_next['Time'].str.startswith('15:30')]
        .set_index('Id')['CumReturnResid'].rename('r1530_next')
    )

    part_a = (r1600_t - r1530_t).rename('part_a')
    part_b = r1530_next.rename('part_b')
    out = pd.DataFrame({'part_a': part_a, 'part_b': part_b}).dropna(how='any')
    out['Target'] = out['part_a'] + out['part_b']
    out['Date'] = int(date_str)
    return out.reset_index()[['Date', 'Id', 'Target', 'part_a', 'part_b']]


def build_all_targets(date_strings, intraday_dir=INTRADAY_DIR):
    """Build targets for all consecutive trading day pairs.

    Returns DataFrame with columns: Date, Id, Target, part_a, part_b.
    """
    dfs = []
    for i in tqdm(range(len(date_strings) - 1), desc='Building targets'):
        one = build_target_single_day(date_strings[i], date_strings[i + 1], intraday_dir)
        if one is not None:
            dfs.append(one)
    return pd.concat(dfs, ignore_index=True)


def add_target_pipeline(df, daily_all, prev_date):
    """Apply the full target normalization pipeline in-place.

    Steps:
      1. Merge EST_VOL_prev and MDV_63_prev from the previous trading day
      2. Target_vol_scaled = Target / EST_VOL_prev
      3. Target_ts         = TS z-score per Id (past-only, window=252, min=60)
      4. Target_mad        = cross-sectional ±5 MAD winsorization of Target_ts
      5. Target_model      = cross-sectional z-score of Target_mad
      6. sample_weight     = sqrt(MDV_63_prev)

    Also keeps Target_clip_cs5mad (raw target clipped at ±5 MAD CS) for diagnostics.

    Rows with missing EST_VOL_prev, Target_model, or sample_weight are dropped.
    Returns the cleaned DataFrame.
    """
    dates = df['Date'].unique()
    daily_prev = build_daily_prev(daily_all, dates, prev_date)

    # Drop pre-existing columns to avoid duplicates on re-run
    for col in ['EST_VOL_prev', 'MDV_63_prev']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(daily_prev[['Date', 'Id', 'EST_VOL_prev', 'MDV_63_prev']],
                  on=['Date', 'Id'], how='left')
    df = df[df['EST_VOL_prev'].notna() & (df['EST_VOL_prev'] > 0)].copy()

    df['Target_vol_scaled'] = df['Target'] / df['EST_VOL_prev']
    df['Target_ts'] = zscore_time_series_per_id(df, 'Target_vol_scaled')
    df['Target_mad'] = winsorize_mad(df, 'Target_ts')
    df['Target_model'] = zscore_cross_sectional(df, 'Target_mad')
    df['Target_clip_cs5mad'] = winsorize_mad(df, 'Target')
    df['sample_weight'] = np.sqrt(df['MDV_63_prev'].replace(0, np.nan))

    df = df.dropna(subset=['Target_model', 'sample_weight'])
    return df
