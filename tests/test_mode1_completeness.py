"""Mode 1 / Mode 2 completeness and data-leakage tests (Section 5.1).

Instructor leakage contract (from project_assignment.pdf):
  "We will test for leakage by applying this mode with and without input
   directories containing future raw data and expect the features to be
   identical.  If your feature creation is found to suffer from leakage,
   your out-of-sample R² will be invalidated and points will be deducted."

This file mirrors that exact test (test_run_mode1_leakage_invariant) plus
additional end-to-end correctness checks for both modes.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import ALL_FEATURE_NAMES, normalize_features

# ---------------------------------------------------------------------------
# Intraday timestamps used in the synthetic dataset (full 15-min grid)
# ---------------------------------------------------------------------------
_TIMES = [
    '09:45', '10:00', '10:15', '10:30', '10:45', '11:00',
    '11:15', '11:30', '11:45', '12:00', '12:15', '12:30',
    '12:45', '13:00', '13:15', '13:30', '13:45', '14:00',
    '14:15', '14:30', '14:45', '15:00', '15:15', '15:30',
]
_N_TIMES = len(_TIMES)

# Security IDs used in synthetic tests
_IDS = ['ID001', 'ID002', 'ID003', 'ID004']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_date_sequence(n: int) -> list[int]:
    """Return n consecutive date ints starting at 20100101."""
    return [20100101 + i for i in range(n)]


def _write_synthetic_raw_data(root: str, dates: list[int], n_ids: int = 4, seed: int = 0):
    """Write minimal synthetic daily + intraday CSVs under *root*.

    Directory layout:
        root/daily_data/dat.YYYYMMDD.csv
        root/intraday_data/YYYYMMDD.csv

    The RNG is seeded once and consumed in date order, so any prefix of *dates*
    will produce bit-for-bit identical files if you call this function again
    with the same seed and a longer *dates* list.
    """
    root_p = Path(root)
    daily_dir = root_p / 'daily_data'
    intra_dir = root_p / 'intraday_data'
    daily_dir.mkdir(parents=True, exist_ok=True)
    intra_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    ids = _IDS[:n_ids]

    for d in dates:
        # ----- intraday -----
        rows = []
        for id_ in ids:
            cum_ret = 0.0
            cum_vol = 0.0
            for t in _TIMES:
                cum_ret += rng.normal(0, 0.002)
                cum_vol += abs(rng.normal(50_000, 10_000))
                rows.append({
                    'Date': d,
                    'Time': f'{t}:00.000',
                    'Id': id_,
                    'CumReturnResid': cum_ret,
                    'CumReturnRaw': cum_ret * 1.01,
                    'CumVolume': cum_vol,
                })
        pd.DataFrame(rows).to_csv(intra_dir / f'{d}.csv', index=False)

        # ----- daily -----
        daily_rows = []
        for id_ in ids:
            close = abs(rng.normal(50, 10)) + 10
            daily_rows.append({
                'Date': d,
                'ID': id_,
                'SYMBOL': id_,
                'MIC': 'XNYS',
                'FREE_FLOAT_PERCENTAGE': 100.0,
                'EST_VOL': abs(rng.normal(0.20, 0.05)) + 0.05,
                'MDV_63': abs(rng.normal(5e7, 1e7)),
                'Open':   close * 0.99,
                'High':   close * 1.01,
                'Low':    close * 0.98,
                'Close':  close,
                'Volume': abs(rng.normal(1e6, 2e5)),
                'PxAdjFactor': 1.0,
                'SharesAdjFactor': 1.0,
            })
        pd.DataFrame(daily_rows).to_csv(daily_dir / f'dat.{d}.csv', index=False)


def _make_feature_panel(dates: list[int], n_ids: int = 4, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic in-memory panel with all 16 feature columns."""
    rng = np.random.default_rng(seed)
    ids = _IDS[:n_ids]
    rows = []
    for d in dates:
        for id_ in ids:
            row = {'Date': d, 'Id': id_}
            for col in ALL_FEATURE_NAMES:
                row[col] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: unit-level leakage invariant (normalize_features)
# ---------------------------------------------------------------------------

def test_normalize_features_future_dates_do_not_change_past():
    """normalize_features: adding future rows must not change past feature values.

    Directly mirrors the instructor's leakage contract at the function level:
    feature values for dates in the target window are identical regardless of
    whether future dates are included in the input panel.
    """
    # Need ≥84 dates for daily features to have non-NaN TS z-scores (see warmup note
    # in test_run_mode1_leakage_invariant).  Use 100 total; target slice 88..97.
    all_dates = _make_date_sequence(100)
    base_dates = all_dates[:95]   # D0..D94
    target_slice = slice(88, 95)  # D88..D94 (present in both)

    panel_base = _make_feature_panel(base_dates)
    panel_full = _make_feature_panel(all_dates)  # same base rows + 5 future rows

    norm_base, _ = normalize_features(panel_base)
    norm_full, _ = normalize_features(panel_full)

    target_base = (
        norm_base[norm_base['Date'].isin(base_dates[target_slice])]
        .sort_values(['Date', 'Id'])
        .reset_index(drop=True)
    )
    target_full = (
        norm_full[norm_full['Date'].isin(base_dates[target_slice])]
        .sort_values(['Date', 'Id'])
        .reset_index(drop=True)
    )

    assert len(target_base) > 0, 'No rows in target date range after normalization'
    pd.testing.assert_frame_equal(
        target_base, target_full,
        atol=1e-10,
        check_like=True,
        obj='normalize_features leakage check',
    )


# ---------------------------------------------------------------------------
# Test 2: end-to-end Mode 1 leakage invariant (mirrors instructor's test)
# ---------------------------------------------------------------------------

def test_run_mode1_leakage_invariant():
    """run_mode1: feature CSVs for target dates are identical with and without future data.

    This is the exact leakage test described in the project spec:
      'We will test for leakage by applying this mode with and without input
       directories containing future raw data and expect the features to be
       identical.'

    Setup:
      - base   dir: D0..D94  (95 dates)
      - future dir: D0..D99  (100 dates — same files for D0..D94, plus D95..D99)
    Both runs request features for D88..D93.  Output CSVs must be identical.

    Note on warmup: daily features (ShortTermReversal etc.) are NaN for indices < 23
    (insufficient lookback); the TS z-score then needs 60 non-NaN past values.
    First fully-normalized date = index 23 + 1(shift) + 60(min_periods) = 84.
    Targets at indices 88–93 are safely past the warmup boundary.
    """
    from main import run_mode1

    all_dates = _make_date_sequence(100)
    base_dates = all_dates[:95]   # D0..D94
    target_start = str(all_dates[88])   # D88
    target_end   = str(all_dates[93])   # D93
    target_dates = all_dates[88:94]

    with tempfile.TemporaryDirectory() as base_in, \
         tempfile.TemporaryDirectory() as future_in, \
         tempfile.TemporaryDirectory() as base_out, \
         tempfile.TemporaryDirectory() as future_out:

        _write_synthetic_raw_data(base_in,   base_dates, seed=7)
        _write_synthetic_raw_data(future_in, all_dates,  seed=7)  # same seed → D0..D94 identical

        run_mode1(base_in,   base_out,   target_start, target_end)
        run_mode1(future_in, future_out, target_start, target_end)

        base_files   = sorted(Path(base_out).glob('*.csv'))
        future_files = sorted(Path(future_out).glob('*.csv'))

        assert len(base_files) > 0, 'run_mode1 wrote no output files (base run)'
        assert len(base_files) == len(future_files), (
            f'base wrote {len(base_files)} files, future wrote {len(future_files)}'
        )

        for bf, ff in zip(base_files, future_files):
            assert bf.name == ff.name, f'File name mismatch: {bf.name} vs {ff.name}'
            df_base   = pd.read_csv(bf)
            df_future = pd.read_csv(ff)
            pd.testing.assert_frame_equal(
                df_base.sort_values('Id').reset_index(drop=True),
                df_future.sort_values('Id').reset_index(drop=True),
                atol=1e-10,
                check_like=True,
                obj=f'leakage check for {bf.name}',
            )


# ---------------------------------------------------------------------------
# Test 3: Mode 1 output columns
# ---------------------------------------------------------------------------

def test_run_mode1_output_columns():
    """run_mode1 output CSVs must have Id + all 16 feature columns, but no Date column."""
    from main import run_mode1

    dates = _make_date_sequence(95)
    target_start = str(dates[88])
    target_end   = str(dates[93])

    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        _write_synthetic_raw_data(raw_dir, dates, seed=11)
        run_mode1(raw_dir, out_dir, target_start, target_end)

        csv_files = sorted(Path(out_dir).glob('*.csv'))
        assert len(csv_files) > 0, 'No output files written'

        df = pd.read_csv(csv_files[0])

        assert 'Id' in df.columns, "Output CSV missing 'Id' column"
        assert 'Date' not in df.columns, (
            "Output CSV must NOT contain 'Date' column (graders expect it absent)"
        )
        missing = [f for f in ALL_FEATURE_NAMES if f not in df.columns]
        assert not missing, f'Output CSV is missing features: {missing}'


# ---------------------------------------------------------------------------
# Test 4: Mode 1 date range filtering
# ---------------------------------------------------------------------------

def test_run_mode1_only_writes_target_date_range():
    """run_mode1 must write exactly the files for dates in [start_date, end_date]."""
    from main import run_mode1

    dates = _make_date_sequence(95)
    target_start = str(dates[88])
    target_end   = str(dates[93])
    expected_names = {f'{d}.csv' for d in dates[88:94]}

    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as out_dir:

        _write_synthetic_raw_data(raw_dir, dates, seed=13)
        run_mode1(raw_dir, out_dir, target_start, target_end)

        written = {p.name for p in Path(out_dir).glob('*.csv')}
        # expected: 6 files for D88..D93
        assert written == expected_names, (
            f'Expected files: {sorted(expected_names)}\n'
            f'Actual files:   {sorted(written)}'
        )


# ---------------------------------------------------------------------------
# Helper: build a self-consistent temporary saved_model directory
# ---------------------------------------------------------------------------

def _make_temp_model_dir(tmpdir: str, feature_cols: list[str]) -> str:
    """Write a minimal, self-consistent set of model artifacts to *tmpdir*.

    Uses a simple Ridge regressor so tests do not depend on the real
    saved_model/ artifacts (which may be inconsistent after pipeline refactors).
    Returns the path to the model directory.
    """
    import pickle
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(feature_cols)
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, n))
    y_train = rng.standard_normal(200)

    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)

    model_dir = Path(tmpdir)
    # Only write best_model.pkl (no ensemble_weights.pkl) so predict() uses single model
    with open(model_dir / 'best_model.pkl',    'wb') as f: pickle.dump(model, f)
    with open(model_dir / 'feature_cols.pkl',  'wb') as f: pickle.dump(feature_cols, f)
    with open(model_dir / 'scaler.pkl',        'wb') as f: pickle.dump(scaler, f)
    with open(model_dir / 'fit_target_mode.pkl','wb') as f: pickle.dump('vol_scaled', f)
    return str(model_dir)


# ---------------------------------------------------------------------------
# Test 5: saved_model/ artifacts are self-consistent
# ---------------------------------------------------------------------------

def test_saved_model_artifacts_consistent():
    """All saved_model/ artifacts must agree on the number of features.

    Fails if feature_cols.pkl, scaler.pkl, and the model(s) were saved from
    different pipeline runs (a common error after a refactor).
    """
    import pickle
    from pathlib import Path as P

    model_dir = P('saved_model')
    fc   = pickle.load(open(model_dir / 'feature_cols.pkl', 'rb'))
    sc   = pickle.load(open(model_dir / 'scaler.pkl',       'rb'))
    n_fc = len(fc)
    n_sc = sc.n_features_in_

    assert n_fc == n_sc, (
        f'feature_cols.pkl has {n_fc} features but scaler.pkl was fitted on {n_sc}. '
        'Re-run the full training pipeline to regenerate consistent artifacts.'
    )

    ew_path = model_dir / 'ensemble_weights.pkl'
    if ew_path.exists():
        xgb_m   = pickle.load(open(model_dir / 'xgb_model.pkl',   'rb'))
        ridge_m  = pickle.load(open(model_dir / 'ridge_model.pkl', 'rb'))
        n_xgb   = xgb_m.n_features_in_
        n_ridge = ridge_m.n_features_in_
        assert n_xgb == n_fc, (
            f'xgb_model.pkl expects {n_xgb} features but feature_cols.pkl has {n_fc}. '
            'Re-run the full training pipeline.'
        )
        assert n_ridge == n_fc, (
            f'ridge_model.pkl expects {n_ridge} features but feature_cols.pkl has {n_fc}. '
            'Re-run the full training pipeline.'
        )
    else:
        best = pickle.load(open(model_dir / 'best_model.pkl', 'rb'))
        n_best = best.n_features_in_
        assert n_best == n_fc, (
            f'best_model.pkl expects {n_best} features but feature_cols.pkl has {n_fc}. '
            'Re-run the full training pipeline.'
        )


# ---------------------------------------------------------------------------
# Test 6: Mode 2 output format (uses a temporary consistent model)
# ---------------------------------------------------------------------------

def test_run_mode2_output_format():
    """run_mode2 output CSVs must have columns Date, Time, Id, Pred with correct values.

    Uses a temporary self-consistent saved model so this test remains independent
    of the actual saved_model/ artifacts (which may be stale after a refactor).
    """
    from main import run_mode2

    feature_cols = ['OvernightReturn', 'LastHourMomentum', 'IntradayReversal',
                    'VolatilityAdjustedReturn']
    dates        = _make_date_sequence(10)
    target_start = str(dates[0])
    target_end   = str(dates[-1])
    ids          = _IDS[:4]
    rng          = np.random.default_rng(17)

    with tempfile.TemporaryDirectory() as feat_dir, \
         tempfile.TemporaryDirectory() as out_dir, \
         tempfile.TemporaryDirectory() as model_dir:

        _make_temp_model_dir(model_dir, feature_cols)

        for d in dates:
            rows = [{'Id': id_, **{col: float(rng.standard_normal()) for col in feature_cols}}
                    for id_ in ids]
            pd.DataFrame(rows).to_csv(Path(feat_dir) / f'{d}.csv', index=False)

        run_mode2(feat_dir, out_dir, model_dir, target_start, target_end)

        csv_files = sorted(Path(out_dir).glob('*.csv'))
        assert len(csv_files) == len(dates), (
            f'Expected {len(dates)} prediction files, got {len(csv_files)}'
        )
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            for col in ('Date', 'Time', 'Id', 'Pred'):
                assert col in df.columns, f"Prediction CSV missing column '{col}'"
            assert (df['Time'] == '15:30:00.000').all(), (
                f"Time column must be '15:30:00.000', got: {df['Time'].unique()}"
            )
            assert set(df['Id']) == set(ids), 'Id values in prediction do not match input'
            assert df['Pred'].notna().all(), 'Prediction CSV contains NaN predictions'


# ---------------------------------------------------------------------------
# Test 7: Mode 2 predictions are deterministic (scaler is transform, not fit_transform)
# ---------------------------------------------------------------------------

def test_run_mode2_predictions_deterministic():
    """run_mode2 must produce identical predictions on two consecutive calls.

    If the scaler were re-fit (fit_transform instead of transform), mean/std would
    differ across calls on different data slices, breaking determinism.
    Uses a temporary consistent model to be independent of saved_model/ artifacts.
    """
    from main import run_mode2

    feature_cols = ['OvernightReturn', 'LastHourMomentum', 'IntradayReversal',
                    'VolatilityAdjustedReturn']
    dates        = _make_date_sequence(10)
    target_start = str(dates[0])
    target_end   = str(dates[-1])
    ids          = _IDS[:4]
    rng          = np.random.default_rng(23)

    with tempfile.TemporaryDirectory() as feat_dir, \
         tempfile.TemporaryDirectory() as out1_dir, \
         tempfile.TemporaryDirectory() as out2_dir, \
         tempfile.TemporaryDirectory() as model_dir:

        _make_temp_model_dir(model_dir, feature_cols)

        for d in dates:
            rows = [{'Id': id_, **{col: float(rng.standard_normal()) for col in feature_cols}}
                    for id_ in ids]
            pd.DataFrame(rows).to_csv(Path(feat_dir) / f'{d}.csv', index=False)

        run_mode2(feat_dir, out1_dir, model_dir, target_start, target_end)
        run_mode2(feat_dir, out2_dir, model_dir, target_start, target_end)

        for d in dates:
            df1 = pd.read_csv(Path(out1_dir) / f'{d}.csv')
            df2 = pd.read_csv(Path(out2_dir) / f'{d}.csv')
            pd.testing.assert_frame_equal(
                df1.sort_values('Id').reset_index(drop=True),
                df2.sort_values('Id').reset_index(drop=True),
                atol=0,
                obj=f'determinism check for {d}.csv',
            )
