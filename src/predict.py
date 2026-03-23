"""Load saved model artifacts and generate predictions on new data."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = 'saved_model'


def load_artifacts(model_dir=MODEL_DIR):
    """Load and return (model, feature_cols, scaler, fit_target_mode) from disk.

    If ensemble artifacts (xgb_model.pkl, ridge_model.pkl, ensemble_weights.pkl)
    exist in model_dir, loads the ensemble instead of the single best_model.
    """
    model_dir = Path(model_dir)
    with open(model_dir / 'feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'fit_target_mode.pkl', 'rb') as f:
        fit_target_mode = pickle.load(f)

    # Prefer ensemble artifacts if present
    if (model_dir / 'ensemble_weights.pkl').exists():
        with open(model_dir / 'xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open(model_dir / 'ridge_model.pkl', 'rb') as f:
            ridge_model = pickle.load(f)
        with open(model_dir / 'ensemble_weights.pkl', 'rb') as f:
            ensemble_weights = pickle.load(f)
        model = {'xgb': xgb_model, 'ridge': ridge_model, 'weights': ensemble_weights}
    else:
        with open(model_dir / 'best_model.pkl', 'rb') as f:
            model = pickle.load(f)

    return model, feature_cols, scaler, fit_target_mode


def predict(df, model_dir=MODEL_DIR, rescale_to_return_space=False):
    """Generate predictions on a new feature DataFrame.

    Args:
        df:                      DataFrame containing at minimum the feature columns.
        model_dir:               path to the directory containing saved .pkl files.
        rescale_to_return_space: if True, multiply normalised predictions by
                                 df['EST_VOL_prev'] to convert back to return space,
                                 as required by the project assignment for holdout
                                 test submission.  Requires 'EST_VOL_prev' in df.
                                 Default False (keeps predictions in Target_model space
                                 for R² evaluation and bias analysis).

    Returns:
        np.ndarray of predictions in Target_model space (default) or return space.
        Uses ensemble (XGB+Ridge blend) if ensemble artifacts exist,
        otherwise uses best_model.pkl.
    """
    model, feature_cols, scaler, _ = load_artifacts(model_dir)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f'Input DataFrame is missing features: {missing}')
    X = scaler.transform(df[feature_cols].astype(float))

    if isinstance(model, dict):
        w_xgb   = model['weights']['xgb']
        w_ridge = model['weights']['ridge']
        raw_preds = w_xgb * model['xgb'].predict(X) + w_ridge * model['ridge'].predict(X)
    else:
        raw_preds = model.predict(X)

    if rescale_to_return_space:
        if 'EST_VOL_prev' not in df.columns:
            raise ValueError("'EST_VOL_prev' required in df when rescale_to_return_space=True")
        return raw_preds * df['EST_VOL_prev'].values
    return raw_preds
