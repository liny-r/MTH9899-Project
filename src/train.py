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
from tqdm import tqdm

from .utils import weighted_r2

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn('XGBoost not installed; XGB model will be skipped. pip install xgboost')

RANDOM_STATE = 42

# --- Hyperparameter grids (edit here to change search ranges) ---
RIDGE_GRID = {
    'alpha': np.logspace(-3, 3, 25).tolist(),
}
RF_GRID = {
    'n_estimators':     [80, 150],
    'max_depth':        [5, 7, 10],
    'min_samples_leaf': [10, 20],
}
XGB_GRID = {
    'max_depth':        [3, 5],
    'learning_rate':    [0.03, 0.1],
    'n_estimators':     [80, 150],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [10, 50],
}
# 2-level full factorial: 2^6 = 64 configs.
# All 6 dimensions explored at low/high values — identical grid on every
# walk-forward fold so R² comparisons are not confounded by search budget.

ELASTICNET_GRID = {
    'alpha':    np.logspace(-5, -0.5, 12).tolist(),
    'l1_ratio': [0.1, 0.5, 0.8, 0.9, 0.95, 1.0],
}


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


def _ridge_one_alpha(alpha, X_train, y_train, w_train, X_val, y_val, w_val):
    """Fit Ridge for one alpha value (used by joblib.Parallel)."""
    m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    m.fit(X_train, y_train, sample_weight=w_train)
    return {'alpha': alpha, 'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val)}


def train_ridge(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train Ridge regression with grid search over alpha.

    Returns (fitted model, val weighted R², results DataFrame).
    """
    scores = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_ridge_one_alpha)(a, X_train, y_train, w_train, X_val, y_val, w_val)
        for a in tqdm(RIDGE_GRID['alpha'], desc='Ridge α', leave=False)
    )
    results = pd.DataFrame(scores)
    best_alpha = results.loc[results['val_weighted_r2'].idxmax(), 'alpha']
    model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def train_random_forest(X_train, y_train, w_train, X_val, y_val, w_val):
    """Train Random Forest with grid search over n_estimators, max_depth, and min_samples_leaf.

    Bootstrap size is capped at 80k rows per tree for speed.
    Each fit uses n_jobs=-1 (built-in tree parallelism); grid search runs sequentially
    to avoid memory-copy overhead of inter-process data transfer.
    Returns (fitted model, val weighted R², results DataFrame).
    """
    n_train = X_train.shape[0]
    max_samples = min(80_000, n_train)
    scores = []
    rf_configs = list(product(RF_GRID['n_estimators'], RF_GRID['max_depth'], RF_GRID['min_samples_leaf']))
    for ne, md, ml in tqdm(rf_configs, desc='RF grid search'):
        m = RandomForestRegressor(
            n_estimators=ne, max_depth=md, min_samples_leaf=ml,
            max_samples=max_samples, random_state=RANDOM_STATE, n_jobs=-1,
        )
        m.fit(X_train, y_train, sample_weight=w_train)
        scores.append({
            'n_estimators': ne, 'max_depth': md, 'min_samples_leaf': ml,
            'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
        })
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = RandomForestRegressor(
        n_estimators=int(best['n_estimators']),
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
    """Train XGBoost with grid search over all XGB_GRID dimensions.

    Returns (fitted model, val weighted R², results DataFrame).
    Returns (None, nan, empty DataFrame) if XGBoost is not installed.
    """
    if not HAS_XGB:
        return None, np.nan, pd.DataFrame()
    scores = []
    xgb_configs = list(product(
        XGB_GRID['max_depth'], XGB_GRID['learning_rate'], XGB_GRID['n_estimators'],
        XGB_GRID['subsample'], XGB_GRID['colsample_bytree'], XGB_GRID['min_child_weight'],
    ))
    for md, lr, ne, sub, col, mcw in tqdm(xgb_configs, desc='XGB grid search'):
        m = xgb.XGBRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=sub, colsample_bytree=col, min_child_weight=mcw,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        m.fit(X_train, y_train, sample_weight=w_train)
        scores.append({
            'max_depth': md, 'learning_rate': lr, 'n_estimators': ne,
            'subsample': sub, 'colsample_bytree': col, 'min_child_weight': mcw,
            'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val),
        })
    results = pd.DataFrame(scores)
    best = results.loc[results['val_weighted_r2'].idxmax()]
    model = xgb.XGBRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        learning_rate=best['learning_rate'],
        subsample=float(best['subsample']),
        colsample_bytree=float(best['colsample_bytree']),
        min_child_weight=int(best['min_child_weight']),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    val_r2 = weighted_r2(y_val, model.predict(X_val), w_val)
    return model, val_r2, results


def _elasticnet_one_config(alpha, l1_ratio, Xt, yt, wt, X_val, y_val, w_val):
    """Fit ElasticNet for one (alpha, l1_ratio) pair (used by joblib.Parallel)."""
    m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
    m.fit(Xt, yt, sample_weight=wt)
    return {'alpha': alpha, 'l1_ratio': l1_ratio,
            'val_weighted_r2': weighted_r2(y_val, m.predict(X_val), w_val)}


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

    en_configs = list(product(ELASTICNET_GRID['alpha'], ELASTICNET_GRID['l1_ratio']))
    scores = Parallel(n_jobs=-1, prefer='processes')(
        delayed(_elasticnet_one_config)(a, l1, Xt, yt, wt, X_val, y_val, w_val)
        for a, l1 in tqdm(en_configs, desc='ElasticNet grid', leave=False)
    )
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
        for cfg in tqdm(grid, desc='MLP grid search', leave=False)
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


def walk_forward_cv(merged_df, feature_cols):
    """Walk-forward cross-validation with expanding training window.

    Trains all 5 models with full hyperparameter grids on each fold.
    Fold structure (expanding window):
        Fold 1: train=2010,          val=2011
        Fold 2: train=2010-2011,     val=2012
        Fold 3: train=2010-2012,     val=2013
        Fold 4: train=2010-2013,     val=2014  ← final fold (replaces train_all_models)

    The final fold's fitted models, scores, results, and scaler are returned
    separately so that downstream cells (best-model selection, ensemble,
    overfitting analysis, white paper) can use them directly — no separate
    train_all_models call is needed.

    Args:
        merged_df:    DataFrame with 'Year' column plus feature_cols, Target_model,
                      and sample_weight. Must span at least 2010-2014.
        feature_cols: ordered list of feature column names.

    Returns:
        wf_results:    DataFrame with columns [fold, train_years, val_year, model, val_r2]
        final_models:  dict of name -> fitted model  (Fold 4)
        final_scores:  dict of name -> val weighted R²  (Fold 4)
        final_results: dict of name -> hyperparameter search DataFrame  (Fold 4)
        final_scaler:  StandardScaler fitted on Fold 4 training data
    """
    folds = [
        ([2010],                2011),
        ([2010, 2011],          2012),
        ([2010, 2011, 2012],    2013),
        ([2010, 2011, 2012, 2013], 2014),
    ]
    all_rows = []
    final_models = final_scores = final_results = final_scaler = None

    for train_years, val_year in folds:
        fold_label = f"{train_years[0]}-{train_years[-1]} → {val_year}"
        print(f'\n--- Fold: {fold_label} ---')
        tr = merged_df[merged_df['Year'].isin(train_years)].copy()
        va = merged_df[merged_df['Year'] == val_year].copy()
        if len(tr) == 0 or len(va) == 0:
            print(f'  Skipping fold {fold_label}: empty train or val.')
            continue

        X_tr, y_tr, w_tr, X_va, y_va, w_va, scaler = prepare_matrices(tr, va, feature_cols)

        fold_models, fold_scores, fold_results = train_all_models(
            X_tr, y_tr, w_tr, X_va, y_va, w_va
        )

        for model_name, r2 in fold_scores.items():
            all_rows.append({
                'fold':        fold_label,
                'train_years': f"{train_years[0]}-{train_years[-1]}",
                'val_year':    val_year,
                'model':       model_name,
                'val_r2':      r2,
            })

        # Capture the final fold's artifacts for downstream use
        if val_year == 2014:
            final_models  = fold_models
            final_scores  = fold_scores
            final_results = fold_results
            final_scaler  = scaler

    return pd.DataFrame(all_rows), final_models, final_scores, final_results, final_scaler


def train_ensemble(models, scores, X_val, y_val, w_val):
    """Blend XGBoost and Ridge predictions via two weighting schemes.

    Evaluates:
        - Equal-weight blend:   0.5 * XGB + 0.5 * Ridge
        - Val-R²-weighted blend: R2_xgb/(R2_xgb+R2_ridge) * XGB + ...

    Args:
        models: dict of name -> fitted model (must contain 'XGBoost' and 'Ridge')
        scores: dict of name -> val weighted R²
        X_val, y_val, w_val: validation arrays

    Returns:
        ensemble_preds: dict of blend_name -> np.ndarray of val predictions
        ensemble_r2:    dict of blend_name -> val weighted R²
        best_blend:     name of the best-performing blend
        best_preds:     prediction array for the best blend
    """
    xgb_pred   = models['XGBoost'].predict(X_val)
    ridge_pred = models['Ridge'].predict(X_val)

    r2_xgb   = scores['XGBoost']
    r2_ridge = scores['Ridge']

    # Equal-weight blend
    equal_preds = 0.5 * xgb_pred + 0.5 * ridge_pred

    # Val-R²-weighted blend
    total = r2_xgb + r2_ridge
    w_xgb   = r2_xgb   / total
    w_ridge = r2_ridge / total
    weighted_preds = w_xgb * xgb_pred + w_ridge * ridge_pred

    ensemble_preds = {
        'XGB+Ridge (equal)':    equal_preds,
        'XGB+Ridge (R²-wtd)':   weighted_preds,
    }
    ensemble_r2 = {name: weighted_r2(y_val, p, w_val) for name, p in ensemble_preds.items()}

    best_blend = max(ensemble_r2, key=ensemble_r2.__getitem__)
    best_preds = ensemble_preds[best_blend]
    return ensemble_preds, ensemble_r2, best_blend, best_preds
