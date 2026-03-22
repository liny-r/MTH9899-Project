"""Model training, evaluation, and artifact persistence."""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from itertools import product
from joblib import Parallel, delayed

from .utils import weighted_r2

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn('XGBoost not installed; XGB model will be skipped. pip install xgboost')

RANDOM_STATE = 42


def prepare_matrices(train_df, val_df, feature_cols):
    """Standardize features and extract X/y/w arrays for train and validation.

    Scaler is fit only on train, then applied to val (no data leakage).

    Returns:
        X_train, y_train, w_train,
        X_val,   y_val,   w_val,
        scaler   (fitted StandardScaler)
    """
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            'Train or validation is empty. '
            'Ensure data covers 2010–2013 (train) and 2014 (validation).'
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].astype(float))
    X_val = scaler.transform(val_df[feature_cols].astype(float))

    y_train = train_df['Target_model'].values
    y_val = val_df['Target_model'].values
    w_train = train_df['sample_weight'].values
    w_val = val_df['sample_weight'].values
    return X_train, y_train, w_train, X_val, y_val, w_val, scaler


def train_ridge(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train Ridge regression with grid search over alpha.

    Returns (fitted model, val weighted R², results DataFrame).
    """
    alphas = np.logspace(-3, 3, 25)
    scores = []
    for alpha in alphas:
        m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        m.fit(X_train, y_train, sample_weight=w_train)
        scores.append({'alpha': alpha, 'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val)})
    results = pd.DataFrame(scores)
    best_alpha = results.loc[results['val_weighted_r2'].idxmax(), 'alpha']
    model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def train_random_forest(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train Random Forest with grid search over max_depth and min_samples_leaf.

    Bootstrap size is capped at 80k rows per tree for speed.
    Returns (fitted model, val weighted R², results DataFrame).
    """
    n_train = X_train.shape[0]
    max_samples = min(80_000, n_train)
    scores = []
    for md in [5, 8]:
        for ml in [10, 20]:
            m = RandomForestRegressor(
                n_estimators=35, max_depth=md, min_samples_leaf=ml,
                max_samples=max_samples, random_state=RANDOM_STATE, n_jobs=-1,
            )
            m.fit(X_train, y_train, sample_weight=w_train)
            scores.append({
                'max_depth': md, 'min_samples_leaf': ml,
                'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
            })
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=int(best['max_depth']),
        min_samples_leaf=int(best['min_samples_leaf']),
        max_samples=max_samples,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def train_xgboost(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train XGBoost with grid search over max_depth, learning_rate, n_estimators.

    Returns (fitted model, val weighted R², results DataFrame).
    Returns (None, nan, empty DataFrame) if XGBoost is not installed.
    """
    if not HAS_XGB:
        return None, np.nan, pd.DataFrame()
    scores = []
    for md in [3, 4, 5, 6]:
        for lr in [0.03, 0.05, 0.1]:
            for ne in [100, 200]:
                m = xgb.XGBRegressor(
                    n_estimators=ne, max_depth=md, learning_rate=lr,
                    random_state=RANDOM_STATE, n_jobs=-1,
                )
                m.fit(X_train, y_train, sample_weight=w_train)
                scores.append({
                    'max_depth': md, 'learning_rate': lr, 'n_estimators': ne,
                    'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
                })
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = xgb.XGBRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=best['learning_rate'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def train_elasticnet(X_train, y_train, w_train, X_val, y_val, w_val, tune_max_rows=100_000):
    """Train ElasticNet with grid search over alpha and l1_ratio.

    Tunes on a random subsample for speed; refits best config on full train.
    Returns (fitted model, val weighted R², results DataFrame).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_train = len(X_train)
    if tune_max_rows is not None and n_train > tune_max_rows:
        idx = rng.choice(n_train, size=tune_max_rows, replace=False)
        Xt, yt, wt = X_train[idx], y_train[idx], w_train[idx]
    else:
        Xt, yt, wt = X_train, y_train, w_train

    alphas = np.logspace(-5, -0.5, 12)
    l1_ratios = [0.8, 0.9, 0.95, 1.0]
    scores = []
    for alpha in alphas:
        for l1 in l1_ratios:
            m = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=RANDOM_STATE)
            m.fit(Xt, yt, sample_weight=wt)
            scores.append({
                'alpha': alpha, 'l1_ratio': l1,
                'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
            })
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = ElasticNet(alpha=best['alpha'], l1_ratio=best['l1_ratio'], random_state=RANDOM_STATE)
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def _mlp_one_config(h_a_lr, Xm, ym, wm, X_val, y_val, w_val, tune_max_iter, batch_size):
    """Train a single MLP configuration (used by joblib.Parallel)."""
    h, a, lr0 = h_a_lr
    m = MLPRegressor(
        hidden_layer_sizes=h, alpha=a, activation='relu', solver='adam',
        batch_size=batch_size, learning_rate='adaptive', learning_rate_init=lr0,
        max_iter=tune_max_iter, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=12, tol=5e-4, random_state=RANDOM_STATE,
    )
    try:
        m.fit(Xm, ym, sample_weight=wm)
    except TypeError:
        m.fit(Xm, ym)
    return {
        'hidden_layer_sizes': h, 'alpha': a, 'learning_rate_init': lr0,
        'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
        'n_iter': getattr(m, 'n_iter_', None),
    }


def train_mlp(X_train, y_train, w_train, X_val, y_val, w_val,
              tune_max_rows=45_000, batch_size=1024):
    """Train MLP with parallel grid search; refit best config on full train.

    Returns (fitted model, val weighted R², results DataFrame).
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)
    n_train = len(X_train)
    if tune_max_rows is not None and n_train > tune_max_rows:
        idx = rng.choice(n_train, size=tune_max_rows, replace=False)
        Xm, ym, wm = X_train[idx], y_train[idx], w_train[idx]
    else:
        Xm, ym, wm = X_train, y_train, w_train

    grid = list(product(
        [(64,), (128,), (128, 64)],   # hidden_layer_sizes
        [1e-4, 1e-3, 1e-2],           # alpha
        [1e-3, 2e-3],                 # learning_rate_init
    ))
    scores = Parallel(n_jobs=-1, prefer='processes')(
        delayed(_mlp_one_config)(cfg, Xm, ym, wm, X_val, y_val, w_val, 250, batch_size)
        for cfg in grid
    )
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = MLPRegressor(
        hidden_layer_sizes=best['hidden_layer_sizes'],
        alpha=float(best['alpha']),
        activation='relu', solver='adam', batch_size=batch_size,
        learning_rate='adaptive', learning_rate_init=float(best['learning_rate_init']),
        max_iter=800, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, tol=1e-4, random_state=RANDOM_STATE,
    )
    try:
        model.fit(X_train, y_train, sample_weight=w_train)
    except TypeError:
        model.fit(X_train, y_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def train_all_models(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train all five models and return a summary dict.

    Returns:
        models:  dict of name -> fitted model
        scores:  dict of name -> val weighted R²
        results: dict of name -> hyperparameter search results DataFrame
    """
    models, scores, results = {}, {}, {}

    print('Training Ridge...')
    models['Ridge'], scores['Ridge'], results['Ridge'] = train_ridge(
        X_train, y_train, w_train, X_val, y_val, w_val)
    print(f"  Ridge val weighted R² = {scores['Ridge']:.6f}")

    print('Training Random Forest...')
    models['Random Forest'], scores['Random Forest'], results['Random Forest'] = train_random_forest(
        X_train, y_train, w_train, X_val, y_val, w_val)
    print(f"  Random Forest val weighted R² = {scores['Random Forest']:.6f}")

    print('Training XGBoost...')
    models['XGBoost'], scores['XGBoost'], results['XGBoost'] = train_xgboost(
        X_train, y_train, w_train, X_val, y_val, w_val)
    print(f"  XGBoost val weighted R² = {scores['XGBoost']:.6f}")

    print('Training ElasticNet...')
    models['ElasticNet'], scores['ElasticNet'], results['ElasticNet'] = train_elasticnet(
        X_train, y_train, w_train, X_val, y_val, w_val)
    print(f"  ElasticNet val weighted R² = {scores['ElasticNet']:.6f}")

    print('Training MLP...')
    models['MLP'], scores['MLP'], results['MLP'] = train_mlp(
        X_train, y_train, w_train, X_val, y_val, w_val)
    print(f"  MLP val weighted R² = {scores['MLP']:.6f}")

    return models, scores, results


def select_best_model(models, scores):
    """Return (best_name, best_model) based on highest val weighted R²."""
    valid = {k: v for k, v in scores.items() if np.isfinite(v) and models.get(k) is not None}
    best_name = max(valid, key=valid.__getitem__)
    return best_name, models[best_name]


def save_artifacts(model_dir, best_model, feature_cols, scaler, fit_target_mode='vol_scaled'):
    """Persist the best model, feature list, scaler, and target mode to disk."""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / 'best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(model_dir / 'feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_dir / 'fit_target_mode.pkl', 'wb') as f:
        pickle.dump(fit_target_mode, f)
    print(f'Saved model artifacts to {model_dir}/')
