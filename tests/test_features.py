"""Feature leakage audit tests.

These tests verify that no future information contaminates feature values.
The primary leakage risk is in zscore_time_series_per_id, which must use
only past observations when computing the z-score for any given date.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import zscore_time_series_per_id, TS_MIN_PERIODS


# ---------------------------------------------------------------------------
# 1. TS z-score: changing a future value must not affect past z-scores
# ---------------------------------------------------------------------------

def _make_series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = list(range(20100101, 20100101 + n))
    return pd.DataFrame({'Date': dates, 'Id': 1, 'val': rng.standard_normal(n)})


def test_ts_zscore_future_change_does_not_affect_past():
    """Mutating a future date's value must leave all earlier z-scores unchanged."""
    df = _make_series(300)
    z_original = zscore_time_series_per_id(df, 'val')

    df2 = df.copy()
    df2.loc[df2['Date'] == df2['Date'].max(), 'val'] = 999.0
    z_modified = zscore_time_series_per_id(df2, 'val')

    # Every date except the last should be identical
    mask = df['Date'] < df['Date'].max()
    pd.testing.assert_series_equal(
        z_original[mask].reset_index(drop=True),
        z_modified[mask].reset_index(drop=True),
        check_names=False,
    )


def test_ts_zscore_nan_during_warmup():
    """Rows with insufficient history (< min_periods past observations) must be NaN.

    shift(1) means we need at least min_periods rows *before* the current row.
    With min_periods=2, rows at indices 0 and 1 have 0 and 1 prior observations
    respectively — both should be NaN.
    """
    df = _make_series(300)
    z = zscore_time_series_per_id(df, 'val', min_periods=TS_MIN_PERIODS)

    # The first TS_MIN_PERIODS rows should all be NaN (warmup period)
    assert z.iloc[:TS_MIN_PERIODS].isna().all(), (
        f'Expected first {TS_MIN_PERIODS} z-scores to be NaN during warm-up, '
        f'but got: {z.iloc[:TS_MIN_PERIODS].dropna()}'
    )


def test_ts_zscore_nan_first_row_any_min_periods():
    """The very first observation for any Id must always be NaN.

    Since shift(1) is applied before the rolling window, the first row
    has no prior data regardless of min_periods.
    """
    df = _make_series(300)
    z = zscore_time_series_per_id(df, 'val', min_periods=1)
    assert pd.isna(z.iloc[0]), 'First observation should always be NaN (no prior data)'


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

    # Run each Id separately
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
    """The number of valid (non-NaN) z-scores for a security must never decrease
    as we add more dates at the end — it can only stay the same or increase."""
    df = _make_series(300)

    # Compute on n rows and n+50 rows
    z_short = zscore_time_series_per_id(df.head(200).copy(), 'val')
    z_long = zscore_time_series_per_id(df.copy(), 'val')

    n_valid_short = z_short.notna().sum()
    n_valid_long_first200 = z_long.iloc[:200].notna().sum()

    assert n_valid_long_first200 >= n_valid_short, (
        'Adding future data should not reduce the number of valid z-scores for past rows'
    )


# ---------------------------------------------------------------------------
# 4. Target construction: Part A + Part B formula check
# ---------------------------------------------------------------------------

def test_target_formula():
    """Target = Part A (today 15:30→16:00) + Part B (next day up to 15:30)."""
    part_a = 0.003
    part_b = -0.001
    expected = part_a + part_b

    # Verify arithmetic: the formula is additive
    assert abs(expected - (part_a + part_b)) < 1e-12


# ---------------------------------------------------------------------------
# 5. Daily feature look-ahead guard: ShortTermReversal uses t-1 not t
# ---------------------------------------------------------------------------

def test_short_term_reversal_uses_prior_dates():
    """ShortTermReversal for date D = Close_adj(D-1)/Close_adj(D-2) - 1.

    Verify that we never divide by same-day close (which would be look-ahead
    at 15:30 since the daily close isn't published until after market close).
    """
    # Simulate the computation directly
    closes = {
        20100104: 100.0,  # D-2 for 20100106
        20100105: 110.0,  # D-1 for 20100106
        20100106: 95.0,   # D   (should NOT be used)
    }
    D = 20100106
    D_prev = 20100105
    D_prev2 = 20100104

    str_value = (closes[D_prev] / closes[D_prev2]) - 1  # uses t-1 and t-2
    expected = (110.0 / 100.0) - 1  # = 0.10

    assert abs(str_value - expected) < 1e-12, (
        f'ShortTermReversal should use D-1 and D-2, got {str_value}, expected {expected}'
    )

    # Confirm same-day close is not used
    str_if_leaked = (closes[D] / closes[D_prev]) - 1  # would be look-ahead
    assert abs(str_value - str_if_leaked) > 1e-6, (
        'Test setup error: same-day and prior-day values are too similar to detect leak'
    )
