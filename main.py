"""main.py — MTH9899 Project CLI entry point.

Section 5.1 of the project spec requires a command-line program with two modes:

  Mode 1 — Feature generation:
      python main.py -m 1 -i /path/to/data -o /path/to/features -s 20150101 -e 20151231

      Builds features for all securities between start and end date (inclusive).
      Writes one CSV per date under the output directory with all feature columns.
      The input directory must contain daily and intraday subdirectories.
      Recognised naming conventions (tried in order):
        daily_data/   + intraday_data/   (OOS / holdout format)
        DailyData/    + data_intraday/   (training format)

  Mode 2 — Prediction:
      python main.py -m 2 -i /path/to/features -o /path/to/preds -p saved_model -s 20150101 -e 20151231

      Loads feature CSVs produced by Mode 1, applies the saved model, and writes
      prediction CSVs (columns: Date, Time, Id, Pred) under the output directory.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import build_date_maps, load_daily_data
from src.features import (
    build_intraday_features,
    build_daily_features,
    attach_daily_prev,
    normalize_features,
)
from src.predict import predict


def parse_args():
    parser = argparse.ArgumentParser(
        description='MTH9899 stock-return prediction pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-i', required=True, metavar='INPUT_DIR',
                        help='Mode 1: parent of DailyData/ and data_intraday/. '
                             'Mode 2: directory of feature CSV files.')
    parser.add_argument('-o', required=True, metavar='OUTPUT_DIR',
                        help='Directory for output files (created if missing).')
    parser.add_argument('-p', default='saved_model', metavar='MODEL_DIR',
                        help='Directory containing saved model artifacts '
                             '(default: saved_model/). Mode 2 only.')
    parser.add_argument('-s', required=True, metavar='YYYYMMDD',
                        help='Start date (inclusive).')
    parser.add_argument('-e', required=True, metavar='YYYYMMDD',
                        help='End date (inclusive).')
    parser.add_argument('-m', required=True, type=int, choices=[1, 2], metavar='MODE',
                        help='1 = feature generation; 2 = prediction.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mode 1 — Feature generation
# ---------------------------------------------------------------------------

def run_mode1(input_dir, output_dir, start_date, end_date):
    """Build and save features for each trading date in [start_date, end_date].

    Loads ALL available data up to end_date so that the 252-day rolling TS
    z-score has a full lookback window.  Only dates in [start_date, end_date]
    are written to disk, ensuring the anti-leakage property: the feature values
    for any date D are identical regardless of whether future data beyond D is
    present in input_dir.
    """
    # Auto-detect subdirectory naming convention
    root = Path(input_dir)
    _CANDIDATES = [
        ('daily_data', 'intraday_data'),   # OOS / holdout format
        ('DailyData',  'data_intraday'),   # training format
    ]
    daily_dir = intraday_dir = None
    for d_name, i_name in _CANDIDATES:
        if (root / d_name).exists() and (root / i_name).exists():
            daily_dir    = root / d_name
            intraday_dir = root / i_name
            break
    if daily_dir is None:
        tried = ' or '.join(f'{d}/{i}' for d, i in _CANDIDATES)
        sys.exit(f'ERROR: Could not find daily+intraday subdirs under {root}. Tried: {tried}')

    s = int(start_date)
    e = int(end_date)

    print(f'[Mode 1] Building features for {start_date}–{end_date}')
    print(f'  daily_dir    = {daily_dir}')
    print(f'  intraday_dir = {intraday_dir}')

    # Build date maps from all available intraday files, then cap at end_date
    date_strings, date_list, prev_date = build_date_maps(str(intraday_dir))
    date_strings_avail = [d for d in date_strings if int(d) <= e]
    date_list_avail    = [int(d) for d in date_strings_avail]

    if not date_strings_avail:
        sys.exit(f'ERROR: No intraday files found on or before {end_date}')

    # Load daily data (cap at end_date to avoid any look-ahead in daily fields)
    print('  Loading daily data...')
    daily_all = load_daily_data(str(daily_dir))
    daily_all = daily_all[daily_all['Date'] <= e]

    # Build features using all available history (TS z-score needs lookback)
    print('  Building intraday features...')
    feat_df = build_intraday_features(date_strings_avail, date_list_avail, str(intraday_dir))

    print('  Adding daily features...')
    feat_df = build_daily_features(feat_df, daily_all, date_list_avail, prev_date)
    feat_df = attach_daily_prev(feat_df, daily_all, prev_date)

    print('  Normalizing features...')
    feat_df, _ = normalize_features(feat_df)

    # Write one CSV per date in [start_date, end_date]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = feat_df[(feat_df['Date'] >= s) & (feat_df['Date'] <= e)]
    n_written = 0
    for date_int, grp in target_dates.groupby('Date'):
        out_path = out_dir / f'{date_int}.csv'
        grp.drop(columns=['Date'], errors='ignore').to_csv(out_path, index=False)
        n_written += 1

    print(f'[Mode 1] Done. Wrote {n_written} feature files to {out_dir}')


# ---------------------------------------------------------------------------
# Mode 2 — Prediction
# ---------------------------------------------------------------------------

def run_mode2(feature_dir, output_dir, model_dir, start_date, end_date):
    """Load feature CSVs and generate prediction CSVs for each date in [start_date, end_date].

    Output CSV columns: Date, Time, Id, Pred.
    Predictions are in normalized Target_model space, which is the space used
    by the graders for R² evaluation.
    """
    s = int(start_date)
    e = int(end_date)
    feat_dir = Path(feature_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(feat_dir.glob('*.csv'))
    if not csv_files:
        sys.exit(f'ERROR: No feature CSV files found in {feat_dir}')

    print(f'[Mode 2] Generating predictions for {start_date}–{end_date}')
    print(f'  feature_dir = {feat_dir}')
    print(f'  model_dir   = {model_dir}')

    n_written = 0
    for csv_path in csv_files:
        try:
            date_int = int(csv_path.stem)
        except ValueError:
            continue
        if not (s <= date_int <= e):
            continue

        feat_df = pd.read_csv(csv_path)
        preds   = predict(feat_df, model_dir=str(model_dir))

        pd.DataFrame({
            'Date': date_int,
            'Time': '15:30:00.000',
            'Id':   feat_df['Id'],
            'Pred': preds,
        }).to_csv(out_dir / f'{date_int}.csv', index=False)
        n_written += 1

    print(f'[Mode 2] Done. Wrote {n_written} prediction files to {out_dir}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    if args.m == 1:
        run_mode1(args.i, args.o, args.s, args.e)
    else:
        run_mode2(args.i, args.o, args.p, args.s, args.e)
