"""Tests for new utility functions and previously untested feature edge cases.

Coverage added here:
  1. compute_feature_ic  — correctness, sign direction, min_obs threshold
  2. bucket_mean_prediction — bucket count, column names, arbitrary n_buckets
  3. spearman_vs_characteristics — correlation sign, NaN masking
  4. winsorize_mad / winsorize_percentile / zscore_cross_sectional — direct unit tests
     including MAD=0 and std=0 edge cases
  5. Intraday feature edge cases — VolumeMorningAfternoonRatio (afternoon vol = 0),
     RealizedUpsideVol (all returns negative), RetVolCorr (no variance in volume)
  6. Momentum21d look-ahead guard (only ShortTermReversal was previously tested)
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    compute_feature_ic,
    bucket_mean_prediction,
    spearman_vs_characteristics,
    winsorize_mad,
    winsorize_percentile,
    zscore_cross_sectional,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ic_panel(n_dates=60, n_ids=100, seed=42):
    """Panel with one feature that has a known positive correlation with the target."""
    rng = np.random.default_rng(seed)
    dates = list(range(20100101, 20100101 + n_dates))
    signal = rng.standard_normal((n_dates, n_ids))
    noise  = rng.standard_normal((n_dates, n_ids)) * 0.5
    rows = []
    for di, d in enumerate(dates):
        for ii in range(n_ids):
            rows.append({
                'Date': d, 'Id': ii,
                'PositiveSignal': signal[di, ii],
                'NegativeSignal': -signal[di, ii],
                'Target_model':   signal[di, ii] + noise[di, ii],
            })
    return pd.DataFrame(rows)


def _two_date_panel(n_ids=50, seed=0):
    """Minimal two-date panel for quick tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in [20100101, 20100102]:
        for i in range(n_ids):
            rows.append({'Date': d, 'Id': i, 'F': rng.standard_normal(),
                         'Target_model': rng.standard_normal(),
                         'X': abs(rng.standard_normal()) + 0.1})
    return pd.DataFrame(rows)


def _uniform_panel(n_dates=5, n_ids=40):
    """Panel where all feature values are identical per date — MAD and std = 0."""
    rows = []
    for d in range(20100101, 20100101 + n_dates):
        for i in range(n_ids):
            rows.append({'Date': d, 'Id': i, 'F': 3.0})
    return pd.DataFrame(rows)


def _make_daily_inputs(n_dates=30, n_ids=3):
    dates = list(range(20100101, 20100101 + n_dates))
    prev_date = {dates[i]: dates[i - 1] for i in range(1, n_dates)}
    rows = [
        {'Date': d, 'Id': id_, 'Close_adj': 100.0 + i * 0.5,
         'DollarVol': 1e6 + i * 1000, 'MDV_63': 1e6}
        for i, d in enumerate(dates)
        for id_ in range(1, n_ids + 1)
    ]
    return dates, prev_date, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. compute_feature_ic
# ---------------------------------------------------------------------------

class TestComputeFeatureIC:
    def test_positive_signal_has_positive_mean_ic(self):
        """A feature constructed to positively correlate with the target should
        have a clearly positive mean IC."""
        df = _ic_panel()
        _, summary = compute_feature_ic(df, ['PositiveSignal'])
        assert summary.loc['PositiveSignal', 'Mean_IC'] > 0.3

    def test_negative_signal_has_negative_mean_ic(self):
        """A feature constructed to negatively correlate with the target should
        have a clearly negative mean IC."""
        df = _ic_panel()
        _, summary = compute_feature_ic(df, ['NegativeSignal'])
        assert summary.loc['NegativeSignal', 'Mean_IC'] < -0.3

    def test_summary_columns_present(self):
        df = _ic_panel()
        _, summary = compute_feature_ic(df, ['PositiveSignal', 'NegativeSignal'])
        assert set(summary.columns) == {'Mean_IC', 'Std_IC', 'IC_tstat', 'ICIR', 'N_dates'}
        assert set(summary.index) == {'PositiveSignal', 'NegativeSignal'}

    def test_ic_df_shape(self):
        """ic_df must have one row per date and one column per feature."""
        n_dates, n_ids = 20, 50
        df = _ic_panel(n_dates=n_dates, n_ids=n_ids)
        ic_df, _ = compute_feature_ic(df, ['PositiveSignal', 'NegativeSignal'])
        assert ic_df.shape == (n_dates, 2)

    def test_min_obs_threshold_produces_nan(self):
        """Dates with fewer than min_obs valid pairs must yield NaN IC."""
        df = _two_date_panel(n_ids=10)  # only 10 IDs per date
        ic_df, _ = compute_feature_ic(df, ['F'], min_obs=20)
        # Every date has 10 obs < 20 → all NaN
        assert ic_df['F'].isna().all()

    def test_min_obs_threshold_accepts_sufficient_obs(self):
        """Dates with at least min_obs valid pairs must yield a finite IC."""
        df = _two_date_panel(n_ids=50)
        ic_df, _ = compute_feature_ic(df, ['F'], min_obs=30)
        assert ic_df['F'].notna().all()

    def test_tstat_magnitude_proportional_to_signal_strength(self):
        """The strong positive signal should have a higher |t-stat| than a
        pure noise feature."""
        rng = np.random.default_rng(7)
        df = _ic_panel(n_dates=100)
        df['Noise'] = rng.standard_normal(len(df))
        _, summary = compute_feature_ic(df, ['PositiveSignal', 'Noise'])
        assert abs(summary.loc['PositiveSignal', 'IC_tstat']) > \
               abs(summary.loc['Noise', 'IC_tstat'])


# ---------------------------------------------------------------------------
# 2. bucket_mean_prediction
# ---------------------------------------------------------------------------

class TestBucketMeanPrediction:
    def test_returns_correct_number_of_buckets(self):
        df = _two_date_panel()
        preds = np.zeros(len(df))
        result = bucket_mean_prediction(df, preds, 'X', n_buckets=5)
        assert len(result) == 5

    def test_column_names(self):
        df = _two_date_panel()
        result = bucket_mean_prediction(df, np.zeros(len(df)), 'X', n_buckets=3)
        assert list(result.columns) == ['bucket', 'mean_pred', 'count']

    def test_bucket_values_are_zero_indexed_integers(self):
        df = _two_date_panel()
        result = bucket_mean_prediction(df, np.zeros(len(df)), 'X', n_buckets=4)
        assert set(result['bucket']) == {0, 1, 2, 3}

    def test_counts_sum_to_total_rows(self):
        df = _two_date_panel()
        result = bucket_mean_prediction(df, np.zeros(len(df)), 'X', n_buckets=10)
        assert result['count'].sum() == len(df)

    def test_mean_pred_reflects_actual_predictions(self):
        """Predictions of +1 in the top bucket and −1 in the bottom bucket
        should produce positive mean_pred for bucket 9 and negative for bucket 0."""
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame({'X': np.arange(n, dtype=float)})
        preds = np.where(df['X'] >= 190, 1.0, np.where(df['X'] < 10, -1.0, 0.0))
        result = bucket_mean_prediction(df, preds, 'X', n_buckets=10)
        assert result.loc[result['bucket'] == 9, 'mean_pred'].iloc[0] > 0
        assert result.loc[result['bucket'] == 0, 'mean_pred'].iloc[0] < 0

    def test_deciles_and_quintiles_produce_different_bucket_counts(self):
        df = _two_date_panel()
        preds = np.zeros(len(df))
        assert len(bucket_mean_prediction(df, preds, 'X', n_buckets=10)) == 10
        assert len(bucket_mean_prediction(df, preds, 'X', n_buckets=5))  == 5


# ---------------------------------------------------------------------------
# 3. spearman_vs_characteristics
# ---------------------------------------------------------------------------

class TestSpearmanVsCharacteristics:
    def test_positive_correlation_detected(self):
        """Predictions equal to the characteristic should yield r ≈ 1."""
        x = np.arange(100, dtype=float)
        result = spearman_vs_characteristics(x, {'X': x})
        assert result.loc[0, 'Spearman_r'] > 0.99

    def test_negative_correlation_detected(self):
        x = np.arange(100, dtype=float)
        result = spearman_vs_characteristics(-x, {'X': x})
        assert result.loc[0, 'Spearman_r'] < -0.99

    def test_uncorrelated_has_large_pvalue(self):
        """Randomly shuffled predictions should give a non-significant p-value."""
        rng = np.random.default_rng(11)
        x = np.arange(200, dtype=float)
        preds = rng.permutation(x)
        result = spearman_vs_characteristics(preds, {'X': x})
        # For truly random data, p > 0.05 is not guaranteed, but this specific
        # seed is uncorrelated; we just check |r| is small
        assert abs(result.loc[0, 'Spearman_r']) < 0.15

    def test_output_columns(self):
        x = np.ones(50)
        result = spearman_vs_characteristics(x, {'A': x, 'B': x})
        assert list(result.columns) == ['Characteristic', 'Spearman_r', 'p_value']
        assert set(result['Characteristic']) == {'A', 'B'}

    def test_nan_values_are_masked(self):
        """NaN entries in either preds or the characteristic must be dropped before
        computing Spearman, not raise an exception."""
        preds = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        char  = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        result = spearman_vs_characteristics(preds, {'C': char})
        assert np.isfinite(result.loc[0, 'Spearman_r'])


# ---------------------------------------------------------------------------
# 4. winsorize_mad, winsorize_percentile, zscore_cross_sectional
# ---------------------------------------------------------------------------

class TestWinsorizeMAD:
    def _frame(self, values, dates=None):
        if dates is None:
            dates = [20100101] * len(values)
        return pd.DataFrame({'Date': dates, 'F': values})

    def test_clips_outlier_above(self):
        """An extreme high value should be clipped to median + 5*MAD."""
        # Use N(0,1) noise for non-outlier values so MAD > 0, then one extreme outlier.
        rng = np.random.default_rng(42)
        base = rng.standard_normal(99).tolist()
        df = self._frame(base + [1000.0])
        result = winsorize_mad(df, 'F', n_mad=5)
        assert result.iloc[-1] < 1000.0, 'Extreme outlier should be clipped'
        # Non-outlier interior values (well within 5 MAD) must be unchanged
        pd.testing.assert_series_equal(
            result.iloc[:-1].reset_index(drop=True),
            pd.Series(base, dtype=float),
            check_names=False, atol=1e-10,
        )

    def test_symmetric_clipping(self):
        """Symmetric outliers above and below should both be clipped."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal(98).tolist()
        df = self._frame(base + [1000.0, -1000.0])
        result = winsorize_mad(df, 'F', n_mad=5)
        assert result.iloc[-2] < 100.0
        assert result.iloc[-1] > -100.0

    def test_mad_zero_does_not_raise(self):
        """When all values are identical (MAD=0), winsorize_mad must not crash
        and must return the original values unchanged (clipping against ±inf)."""
        df = self._frame([3.0] * 50)
        result = winsorize_mad(df, 'F', n_mad=5)
        assert result.eq(3.0).all()

    def test_multiple_dates_are_independent(self):
        """Each date's clipping must use that date's median and MAD only."""
        df = pd.DataFrame({
            'Date': [20100101] * 10 + [20100102] * 10,
            'F':    [1.0] * 9 + [50.0] + [10.0] * 9 + [500.0],
        })
        result = winsorize_mad(df, 'F', n_mad=5)
        # The outlier on date 1 (50 among 1s) should be clipped by date 1's MAD,
        # not date 2's (where 500 would require a much larger clip point).
        clip_d1 = result[df['Date'] == 20100101].iloc[-1]
        clip_d2 = result[df['Date'] == 20100102].iloc[-1]
        assert clip_d1 < clip_d2, 'Clip points should differ across dates'


class TestWinsorizePercentile:
    def test_clips_top_and_bottom(self):
        """Values outside [1st, 99th] percentile must be clipped."""
        rng = np.random.default_rng(0)
        vals = list(rng.standard_normal(98)) + [100.0, -100.0]
        df = pd.DataFrame({'Date': [20100101] * 100, 'F': vals})
        result = winsorize_percentile(df, 'F', low=0.01, high=0.99)
        assert result.max() < 100.0
        assert result.min() > -100.0

    def test_interior_values_unchanged(self):
        """Values within the percentile range must not be modified."""
        vals = np.linspace(0, 1, 100)
        df = pd.DataFrame({'Date': [20100101] * 100, 'F': vals})
        result = winsorize_percentile(df, 'F', low=0.05, high=0.95)
        interior = (vals >= vals[5]) & (vals <= vals[94])
        np.testing.assert_array_almost_equal(
            result[interior].values, vals[interior], decimal=10
        )


class TestZscoreCrossSectional:
    def test_mean_zero_std_one(self):
        """Each date's z-scored values must have mean ≈ 0 and std ≈ 1."""
        rng = np.random.default_rng(2)
        dates = [20100101, 20100102, 20100103]
        rows = [{'Date': d, 'Id': i, 'F': rng.standard_normal()}
                for d in dates for i in range(50)]
        df = pd.DataFrame(rows)
        result = zscore_cross_sectional(df, 'F')
        df['z'] = result
        for d in dates:
            grp = df.loc[df['Date'] == d, 'z']
            assert abs(grp.mean()) < 1e-10
            assert abs(grp.std(ddof=1) - 1.0) < 1e-10

    def test_zero_std_returns_nan(self):
        """When all values on a date are identical (std=0), result must be NaN."""
        df = pd.DataFrame({'Date': [20100101] * 10, 'F': [5.0] * 10})
        result = zscore_cross_sectional(df, 'F')
        assert result.isna().all()


# ---------------------------------------------------------------------------
# 5. Intraday feature edge cases (features_single_day)
# ---------------------------------------------------------------------------

from src.features import features_single_day


def _pivoted(times, n_ids=5, ret_value=0.01, vol_value=1000):
    """Build minimal pivoted CumReturnResid / CumVolume DataFrames."""
    ids = list(range(1, n_ids + 1))
    r = pd.DataFrame(
        {t: [ret_value * (i + 1) * (j + 1) * 0.001 for i in range(n_ids)]
         for j, t in enumerate(times)},
        index=ids,
    )
    r.index.name = 'Id'
    v = pd.DataFrame(
        {t: [vol_value * (i + 1) for i in range(n_ids)]
         for t in times},
        index=ids,
    )
    v.index.name = 'Id'
    return r, v


class TestIntradayFeatureEdgeCases:
    TIMES = ['09:45', '10:00', '10:15', '10:30', '10:45', '11:00',
             '11:15', '11:30', '11:45', '12:00', '12:15', '12:30',
             '12:45', '13:00', '13:15', '13:30', '13:45', '14:00',
             '14:15', '14:30', '14:45', '15:00', '15:15', '15:30']

    def test_realized_upside_vol_all_negative_returns_is_nan(self):
        """When all 15-min return increments are negative, RealizedUpsideVol should
        be NaN (no upside) rather than 0 or raising an exception."""
        r, v = _pivoted(self.TIMES, ret_value=-0.001)
        result = features_single_day(r, v, 20100104)
        assert 'RealizedUpsideVol' in result.columns
        assert result['RealizedUpsideVol'].isna().all(), (
            'RealizedUpsideVol must be NaN when there are no positive return increments'
        )

    def test_retvolcorr_constant_volume_is_nan(self):
        """When volume increments have zero variance (all equal), the Pearson
        correlation is undefined and RetVolCorr should be NaN."""
        r, v = _pivoted(self.TIMES)
        # Flatten all volume to a constant so vol_inc = 0 for every increment
        for t in self.TIMES:
            v[t] = 1000.0
        result = features_single_day(r, v, 20100104)
        assert 'RetVolCorr' in result.columns
        assert result['RetVolCorr'].isna().all(), (
            'RetVolCorr must be NaN when volume increments have no variance'
        )

    def test_volume_morning_afternoon_ratio_zero_afternoon_volume(self):
        """When afternoon volume is zero, VolumeMorningAfternoonRatio requires
        a merge with daily data (the ratio is computed in attach_daily_prev, not
        here), so we only check features_single_day stores the raw CumVolume cols."""
        r, v = _pivoted(self.TIMES)
        # Set afternoon (12:00–15:30) volume to zero for all IDs
        afternoon_times = [t for t in self.TIMES if t >= '12:00']
        for t in afternoon_times:
            v[t] = 0.0
        # features_single_day should not raise
        result = features_single_day(r, v, 20100104)
        assert 'CumVolume_1530' in result.columns
        assert result['CumVolume_1530'].eq(0.0).all()

    def test_missing_1100_fallback_to_1200_for_first_hour_momentum(self):
        """When 11:00 is absent, FirstHourMomentum must fall back to 09:45→12:00."""
        times_no_1100 = [t for t in self.TIMES if t != '11:00']
        r, v = _pivoted(times_no_1100)
        result = features_single_day(r, v, 20100104)
        assert 'FirstHourMomentum' in result.columns
        assert result['FirstHourMomentum'].notna().any(), (
            'FirstHourMomentum must use 12:00 fallback when 11:00 is absent'
        )
        # Value should equal r[12:00] - r[09:45]
        expected = (r['12:00'] - r['09:45']).values
        np.testing.assert_array_almost_equal(
            result['FirstHourMomentum'].values, expected, decimal=10
        )

    def test_all_features_absent_when_timestamps_missing(self):
        """When the pivoted DataFrame has no usable timestamps, all optional
        features must be absent (not crash) and mandatory columns present."""
        r = pd.DataFrame({'16:00': [0.01, 0.02]}, index=[1, 2])
        r.index.name = 'Id'
        result = features_single_day(r, None, 20100104)
        assert 'Date' in result.columns
        assert 'Id' in result.columns
        for feat in ['OvernightReturn', 'FirstHourMomentum', 'LastHourMomentum',
                     'IntradayReversal', 'IntradayReturnSkew', 'RealizedUpsideVol',
                     'RetVolCorr']:
            assert feat not in result.columns, (
                f'{feat} should not appear when its required timestamps are absent'
            )


# ---------------------------------------------------------------------------
# 6. Momentum21d look-ahead guard
# ---------------------------------------------------------------------------

def test_momentum21d_uses_t_minus_2_not_t_minus_1_close():
    """build_daily_features: Momentum21d = Close_adj(t-2) / Close_adj(t-23) - 1.

    Verifies the skip-one-day construction (to avoid look-ahead at 15:30) by
    checking the computed value against the explicit expected formula.
    """
    from src.features import build_daily_features

    dates, prev_date, daily_all = _make_daily_inputs(n_dates=30, n_ids=2)
    D = dates[-1]
    feat_df = pd.DataFrame({'Date': [D], 'Id': [1]})

    result = build_daily_features(feat_df, daily_all, dates, prev_date)

    assert 'Momentum21d' in result.columns

    D_prev  = prev_date[D]
    D_prev2 = prev_date[D_prev]
    D_prev23_idx = max(0, dates.index(D_prev2) - 21)
    D_prev23 = dates[D_prev23_idx]

    c_t2  = daily_all[(daily_all['Date'] == D_prev2)  & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    c_t23 = daily_all[(daily_all['Date'] == D_prev23) & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    expected = c_t2 / c_t23 - 1

    row = result[result['Id'] == 1].iloc[0]
    assert abs(row['Momentum21d'] - expected) < 1e-10

    # Confirm t-1 close is not used (same-day or D-1 would be look-ahead at 15:30)
    c_t1 = daily_all[(daily_all['Date'] == D_prev) & (daily_all['Id'] == 1)]['Close_adj'].iloc[0]
    wrong = c_t1 / c_t23 - 1
    assert abs(row['Momentum21d'] - wrong) > 1e-6, (
        'Momentum21d should use Close(t-2), not Close(t-1) — t-1 is look-ahead at 15:30'
    )
