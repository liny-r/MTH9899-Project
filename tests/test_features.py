"""Feature leakage audit and pipeline correctness tests.

These tests verify:
  1. No future information contaminates feature values (look-ahead leakage).
  2. The actual pipeline functions produce correct outputs on synthetic data.
  3. normalize_features applies the right winsorization per feature category.
  4. add_target_pipeline produces a properly z-scored target.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import zscore_time_series_per_id, TS_MIN_PERIODS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = list(range(20100101, 20100101 + n))
    return pd.DataFrame({'Date': dates, 'Id': 1, 'val': rng.standard_normal(n)})


def _make_daily_features_inputs(n_dates=30, n_ids=5):
    """Build minimal inputs for build_daily_features with monotonically increasing closes."""
    dates = list(range(20100101, 20100101 + n_dates))
    prev_date = {dates[i]: dates[i - 1] for i in range(1, n_dates)}
    rows = [
        {'Date': d, 'Id': id_, 'Close_adj': 100.0 + i, 'DollarVol': 1e6, 'MDV_63': 1e6}
        for i, d in enumerate(dates)
        for id_ in range(1, n_ids + 1)
    ]
    return dates, prev_date, pd.DataFrame(rows)


def _make_feature_panel(n_dates=150, n_ids=60, seed=7):
    """Build a panel DataFrame with all 13 feature columns filled with normal noise."""
    from src.features import ALL_FEATURE_NAMES
    rng = np.random.default_rng(seed)
    dates = list(range(20100101, 20100101 + n_dates))
    rows = []
    for d in dates:
        for id_ in range(1, n_ids + 1):
            row = {'Date': d, 'Id': id_}
            for col in ALL_FEATURE_NAMES:
                row[col] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. TS z-score: no look-ahead
# ---------------------------------------------------------------------------

def test_ts_zscore_future_change_does_not_affect_past():
    """Mutating a future date's value must leave all earlier z-scores unchanged."""
    df = _make_series(300)
    z_original = zscore_time_series_per_id(df, 'val')

    df2 = df.copy()
    df2.loc[df2['Date'] == df2['Date'].max(), 'val'] = 999.0
    z_modified = zscore_time_series_per_id(df2, 'val')

    mask = df['Date'] < df['Date'].max()
    pd.testing.assert_series_equal(
        z_original[mask].reset_index(drop=True),
        z_modified[mask].reset_index(drop=True),
        check_names=False,
    )


def test_ts_zscore_nan_during_warmup():
    """Warmup rows must be NaN; all post-warmup rows must be valid.

    With 300 rows and min_periods=60, exactly the first 60 rows should be NaN
    and all remaining 240 rows should be non-NaN.
    """
    df = _make_series(300)
    z = zscore_time_series_per_id(df, 'val', min_periods=TS_MIN_PERIODS)

    assert z.iloc[:TS_MIN_PERIODS].isna().all(), (
        f'Expected first {TS_MIN_PERIODS} z-scores to be NaN during warm-up, '
        f'but got: {z.iloc[:TS_MIN_PERIODS].dropna()}'
    )
    assert z.iloc[TS_MIN_PERIODS:].notna().all(), (
        f'Expected all {len(z) - TS_MIN_PERIODS} post-warmup z-scores to be valid'
    )


def test_ts_zscore_nan_first_row_any_min_periods():
    """The first two observations must be NaN; all subsequent rows must be valid.

    shift(1) means row 0 has no prior data. A z-score also requires std, which
    needs at least 2 observations, so row 1 (with only 1 prior value) is also NaN.
    With min_periods=2, rows 2 onwards each have at least 2 prior observations
    and must all produce valid z-scores.
    """
    df = _make_series(300)
    z = zscore_time_series_per_id(df, 'val', min_periods=2)
    assert z.iloc[:2].isna().all(), 'First two rows must be NaN (insufficient prior data for std)'
    assert z.iloc[2:].notna().all(), (
        'With min_periods=2, all rows from index 2 onward must produce valid z-scores'
    )


# ---------------------------------------------------------------------------
# 2. TS z-score: multiple securities are handled independently
# ---------------------------------------------------------------------------

def test_ts_zscore_multiple_ids_independent():
    """Each security's z-score must depend only on that security's own past values."""
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'Date': list(range(n)) * 2,
        'Id':   [1] * n + [2] * n,
        'val':  np.concatenate([rng.standard_normal(n), rng.standard_normal(n)]),
    })

    z_together = zscore_time_series_per_id(df, 'val')

    z1 = zscore_time_series_per_id(df[df['Id'] == 1].copy(), 'val')
    z2 = zscore_time_series_per_id(df[df['Id'] == 2].copy(), 'val')

    pd.testing.assert_series_equal(
        z_together[df['Id'] == 1].reset_index(drop=True),
        z1.reset_index(drop=True),
        check_names=False,
        atol=1e-10,
    )
    pd.testing.assert_series_equal(
        z_together[df['Id'] == 2].reset_index(drop=True),
        z2.reset_index(drop=True),
        check_names=False,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 3. TS z-score: result is monotonically informed (more data = more non-NaN)
# ---------------------------------------------------------------------------

def test_ts_zscore_non_nan_count_increases_with_history():
    """Adding N future rows must produce exactly N more valid z-scores.

    The first 200 rows must be unaffected (same non-NaN count), and all 100
    new rows must be valid since they each have far more than min_periods=60
    prior observations.
    """
    df = _make_series(300)

    z_short = zscore_time_series_per_id(df.head(200).copy(), 'val')
    z_long  = zscore_time_series_per_id(df.copy(), 'val')

    n_valid_short          = z_short.notna().sum()
    n_valid_long_first_200 = z_long.iloc[:200].notna().sum()
    n_valid_long_new_100   = z_long.iloc[200:].notna().sum()

    assert n_valid_long_first_200 == n_valid_short, (
        'Adding future rows must not change the non-NaN count of past z-scores'
    )
    assert n_valid_long_new_100 == 100, (
        'All 100 new rows have sufficient history and must produce valid z-scores'
    )


# ---------------------------------------------------------------------------
# 4. Target construction: tests the actual build_target_single_day function
# ---------------------------------------------------------------------------

def test_target_formula_uses_part_a_plus_part_b():
    """build_target_single_day: Target = (r1600-r1530) on D + r1530 on D+1.

    Creates real temporary CSV files and calls the actual pipeline function,
    verifying both part components and their sum.
    """
    from src.target import build_target_single_day

    r1530_today = 0.020
    r1600_today = 0.025
    r1530_next  = -0.005

    today_df = pd.DataFrame({
        'Date': ['20100104', '20100104'],
        'Time': ['15:30:00', '16:00:00'],
        'Id':   [1, 1],
        'CumReturnResid': [r1530_today, r1600_today],
        'CumReturnRaw':   [0.0, 0.0],
        'CumVolume':      [1000, 1100],
    })
    next_df = pd.DataFrame({
        'Date': ['20100105'],
        'Time': ['15:30:00'],
        'Id':   [1],
        'CumReturnResid': [r1530_next],
        'CumReturnRaw':   [0.0],
        'CumVolume':      [900],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        today_df.to_csv(os.path.join(tmpdir, '20100104.csv'), index=False)
        next_df.to_csv(os.path.join(tmpdir, '20100105.csv'), index=False)
        result = build_target_single_day('20100104', '20100105', tmpdir)

    assert result is not None
    row = result[result['Id'] == 1].iloc[0]
    assert abs(row['part_a'] - (r1600_today - r1530_today)) < 1e-10
    assert abs(row['part_b'] - r1530_next) < 1e-10
    assert abs(row['Target'] - (row['part_a'] + row['part_b'])) < 1e-10


# ---------------------------------------------------------------------------
# 5. Daily feature look-ahead guard: tests the actual build_daily_features function
# ---------------------------------------------------------------------------

def test_short_term_reversal_calls_build_daily_features():
    """build_daily_features: ShortTermReversal = Close_adj(t-1)/Close_adj(t-2) - 1.

    Verifies the actual function output uses D-1 and D-2 closes, not the
    same-day close (which would be look-ahead at 15:30).
    """
    from src.features import build_daily_features

    dates, prev_date, daily_all = _make_daily_features_inputs(n_dates=30)
    D = dates[-1]
    feat_df = pd.DataFrame({'Date': [D], 'Id': [1]})

    result = build_daily_features(feat_df, daily_all, dates, prev_date)

    assert 'ShortTermReversal' in result.columns

    D_prev  = prev_date[D]
    D_prev2 = prev_date[D_prev]
    c1 = daily_all[(daily_all['Date'] == D_prev)  & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    c2 = daily_all[(daily_all['Date'] == D_prev2) & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    expected_str = (c1 / c2) - 1

    row = result[result['Id'] == 1].iloc[0]
    assert abs(row['ShortTermReversal'] - expected_str) < 1e-10

    # Confirm same-day close is NOT used
    c_same = daily_all[(daily_all['Date'] == D) & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    assert abs(row['ShortTermReversal'] - ((c_same / c1) - 1)) > 1e-6, (
        'Test setup error or look-ahead detected: same-day close should not be used'
    )


# ---------------------------------------------------------------------------
# 6. normalize_features: correct winsorization per feature category
# ---------------------------------------------------------------------------

def test_normalize_features_produces_all_feature_columns():
    """normalize_features must return all 13 expected feature columns, each
    cross-sectionally z-scored (mean~0, std~1 per date after normalization).
    """
    from src.features import normalize_features, ALL_FEATURE_NAMES

    feat_df = _make_feature_panel()
    norm_df, feature_cols = normalize_features(feat_df)

    assert set(feature_cols) == set(ALL_FEATURE_NAMES)
    assert set(feature_cols).issubset(set(norm_df.columns))

    # Each feature should be cross-sectionally z-scored: mean~0, std~1 per date
    for col in feature_cols:
        per_date = norm_df.groupby('Date')[col].agg(['mean', 'std'])
        valid = per_date[per_date['std'].notna() & (per_date['std'] > 0)]
        assert (valid['mean'].abs() < 1e-6).all(), (
            f'{col}: cross-sectional mean should be ~0 after normalization'
        )
        assert ((valid['std'] - 1.0).abs() < 1e-6).all(), (
            f'{col}: cross-sectional std should be ~1 after normalization'
        )


def test_normalize_features_no_winsor_outlier_more_extreme_than_mad():
    """NO_WINSOR_FEATURES must not be clipped before the cross-sectional z-score.

    An extreme outlier in RetVolCorr (NO_WINSOR) should produce a more extreme
    normalized value than the same outlier in OvernightReturn (MAD), because
    MAD winsorization clips the latter before z-scoring.

    If the bug (applying MAD to all features) were reintroduced, both outliers
    would be clipped to the same MAD bound and this test would fail.
    """
    from src.features import normalize_features, ALL_FEATURE_NAMES

    rng = np.random.default_rng(99)
    n_dates, n_ids = 150, 60
    dates = list(range(20100101, 20100101 + n_dates))
    rows = []
    for d in dates:
        for id_ in range(1, n_ids + 1):
            row = {'Date': d, 'Id': id_}
            for col in ALL_FEATURE_NAMES:
                row[col] = rng.standard_normal()
            rows.append(row)
    feat_df = pd.DataFrame(rows)

    # On the last date inject the same large outlier into both a MAD feature
    # (OvernightReturn) and the NO_WINSOR feature (RetVolCorr) for Id=1
    last_date = dates[-1]
    mask = (feat_df['Date'] == last_date) & (feat_df['Id'] == 1)
    feat_df.loc[mask, 'OvernightReturn'] = 50.0  # MAD feature — gets clipped to ~5 MAD
    feat_df.loc[mask, 'RetVolCorr']      = 50.0  # NO_WINSOR  — passes through

    norm_df, _ = normalize_features(feat_df)
    outlier_row = norm_df[(norm_df['Date'] == last_date) & (norm_df['Id'] == 1)]
    assert len(outlier_row) == 1

    norm_overnight  = abs(float(outlier_row['OvernightReturn'].iloc[0]))
    norm_retvolcorr = abs(float(outlier_row['RetVolCorr'].iloc[0]))

    assert norm_retvolcorr > norm_overnight, (
        f'NO_WINSOR outlier ({norm_retvolcorr:.2f}) should be more extreme than '
        f'MAD-clipped outlier ({norm_overnight:.2f}); check PCT/NO_WINSOR branching'
    )


# ---------------------------------------------------------------------------
# 7. Target pipeline: Target_model must be cross-sectionally z-scored
# ---------------------------------------------------------------------------

def test_target_pipeline_model_is_cross_sectionally_z_scored():
    """add_target_pipeline: Target_model must have mean~0 and std~1 per date.

    This would fail if the CS z-score step were missing (as it was before the
    fix), because the output would then be ±5 MAD-clipped values rather than
    a properly scaled cross-sectional z-score.
    """
    from src.target import add_target_pipeline

    n_dates, n_ids = 120, 80
    rng = np.random.default_rng(5)
    dates = list(range(20100101, 20100101 + n_dates))
    prev_date = {dates[i]: dates[i - 1] for i in range(1, n_dates)}

    df = pd.DataFrame([
        {'Date': d, 'Id': id_, 'Target': rng.standard_normal()}
        for d in dates for id_ in range(1, n_ids + 1)
    ])
    daily_all = pd.DataFrame([
        {'Date': d, 'Id': id_, 'EST_VOL': 0.20, 'MDV_63': 1e6}
        for d in dates for id_ in range(1, n_ids + 1)
    ])

    result = add_target_pipeline(df, daily_all, prev_date)

    per_date = result.groupby('Date')['Target_model'].agg(['mean', 'std'])
    valid = per_date[per_date['std'].notna() & (per_date['std'] > 0)]

    # 1 date dropped (no prev_date for first date) + 60-period TS warmup = 61 dates lost
    expected_valid_dates = n_dates - 1 - TS_MIN_PERIODS
    assert len(valid) == expected_valid_dates, (
        f'Expected {expected_valid_dates} dates with valid Target_model, got {len(valid)}'
    )

    assert (valid['mean'].abs() < 1e-6).all(), (
        'Target_model cross-sectional mean should be 0 after z-scoring; '
        'check that the CS z-score step is applied in add_target_pipeline'
    )
    assert ((valid['std'] - 1.0).abs() < 1e-6).all(), (
        'Target_model cross-sectional std should be 1 after z-scoring; '
        'check that the CS z-score step is applied in add_target_pipeline'
    )
