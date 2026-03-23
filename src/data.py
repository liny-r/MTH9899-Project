"""Data loading utilities for daily and intraday datasets."""

import pandas as pd
import numpy as np
from pathlib import Path

# Default paths relative to the MTH9899-Project/ directory
DAILY_DIR = 'DailyData'
INTRADAY_DIR = 'data_intraday'


def build_date_maps(intraday_dir=INTRADAY_DIR):
    """Scan intraday directory and return date structures.

    Returns:
        date_strings: sorted list of date strings e.g. ['20100104', ...]
        date_list:    sorted list of date ints   e.g. [20100104, ...]
        prev_date:    dict mapping date_int -> previous trading date_int
    """
    data_dir = Path(intraday_dir)
    date_strings = sorted(p.stem for p in data_dir.glob('*.csv'))
    date_list = [int(d) for d in date_strings]
    prev_date = {date_list[i]: date_list[i - 1] for i in range(1, len(date_list))}
    return date_strings, date_list, prev_date


def load_daily_data(daily_dir=DAILY_DIR):
    """Load all daily CSVs and compute adjusted prices and dollar volume.

    Returns a DataFrame with columns:
        Date, Id, EST_VOL, MDV_63, Close, Volume, PxAdjFactor,
        SharesAdjFactor, Close_adj, DollarVol
    """
    daily_dir = Path(daily_dir)
    daily_dfs = []
    for f in sorted(daily_dir.glob('dat.*.csv')):
        date_int = int(f.stem.split('.')[1])  # dat.20100104 -> 20100104
        df = pd.read_csv(f)
        df['Date'] = date_int
        df = df.rename(columns={'ID': 'Id'})
        # Close_adj = Close * PxAdjFactor (comparable across days)
        df['Close_adj'] = df['Close'] * df['PxAdjFactor']
        # DollarVol = adjusted share volume * adjusted close price
        df['DollarVol'] = (df['Volume'] * df['SharesAdjFactor']) * df['Close_adj']
        daily_dfs.append(
            df[['Date', 'Id', 'EST_VOL', 'MDV_63', 'Close', 'Volume',
                'PxAdjFactor', 'SharesAdjFactor', 'Close_adj', 'DollarVol']]
        )
    return pd.concat(daily_dfs, ignore_index=True)


def load_day_pivoted(date_str, intraday_dir=INTRADAY_DIR):
    """Load one day's intraday CSV; return wide DataFrames pivoted by time.

    Returns:
        r: DataFrame of CumReturnResid with columns=time strings, index=Id
        v: DataFrame of CumVolume    with columns=time strings, index=Id
        Returns (None, None) if file does not exist.
    """
    path = Path(intraday_dir) / f'{date_str}.csv'
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df['TimeStr'] = df['Time'].str[:5]  # '09:45:00.000' -> '09:45'
    r = df.pivot_table(index='Id', columns='TimeStr', values='CumReturnResid', aggfunc='first')
    v = df.pivot_table(index='Id', columns='TimeStr', values='CumVolume', aggfunc='first')
    return r, v


def build_daily_prev(daily_all, dates, prev_date):
    """Build a DataFrame with previous-day EST_VOL, MDV_63, Close_adj, and SharesAdjFactor.

    At 15:30 on day D, same-day daily data is not yet available, so we use
    the prior trading day's values.

    Returns DataFrame with columns:
        Date, Id, EST_VOL_prev, MDV_63_prev, Close_adj_prev, SharesAdjFactor_prev
    """
    rows = []
    for D in dates:
        D_prev = prev_date.get(D)
        if D_prev is None:
            continue
        available = ['Id', 'EST_VOL', 'MDV_63']
        for col in ('Close_adj', 'SharesAdjFactor'):
            if col in daily_all.columns:
                available.append(col)
        d = daily_all[daily_all['Date'] == D_prev][available].copy()
        d = d.rename(columns={
            'EST_VOL':         'EST_VOL_prev',
            'MDV_63':          'MDV_63_prev',
            'Close_adj':       'Close_adj_prev',
            'SharesAdjFactor': 'SharesAdjFactor_prev',
        })
        d['Date'] = D
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
