"""Load saved model artifacts and generate predictions on new data."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = 'saved_model'


def load_artifacts(model_dir=MODEL_DIR):
    """Load and return (model, feature_cols, scaler, fit_target_mode) from disk."""
    model_dir = Path(model_dir)
    with open(model_dir / 'best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(model_dir / 'feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'fit_target_mode.pkl', 'rb') as f:
        fit_target_mode = pickle.load(f)
    return model, feature_cols, scaler, fit_target_mode


def predict(df, model_dir=MODEL_DIR):
    """Generate predictions on a new feature DataFrame.

    Args:
        df:        DataFrame containing at minimum the 13 feature columns.
        model_dir: path to the directory containing saved .pkl files.

    Returns:
        np.ndarray of predictions in Target_model space
        (i.e., 5MAD_CS(TS_z(Target / EST_VOL_prev)) space).
    """
    model, feature_cols, scaler, _ = load_artifacts(model_dir)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f'Input DataFrame is missing features: {missing}')
    X = scaler.transform(df[feature_cols].astype(float))
    return model.predict(X)
