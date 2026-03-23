"""Reusable plotting functions for feature analysis, model evaluation, and white-paper figures.

All plotting functions accept an optional ``save_path`` parameter.  When provided,
the figure is saved to that path (PNG/PDF) in addition to being displayed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from IPython.display import display

from .utils import bucket_mean_prediction, spearman_vs_characteristics
from .data import INTRADAY_DIR


# ---------------------------------------------------------------------------
# Feature IC stability
# ---------------------------------------------------------------------------

def ic_stability_plot(ic_df, ic_summary, n_top=4, window=60, save_path=None):
    """Rolling ``window``-day average IC for the top ``n_top`` features.

    Args:
        ic_df:      DataFrame (n_dates × n_features) of daily Spearman ICs.
        ic_summary: DataFrame with Mean_IC column (output of compute_feature_ic).
        n_top:      Number of top features to plot (by |Mean_IC|).
        window:     Rolling window in trading days.
        save_path:  If given, save figure to this path.
    """
    top_feats = ic_summary.head(min(n_top, len(ic_summary))).index.tolist()
    ic_rolling = ic_df[top_feats].rolling(window, min_periods=20).mean()
    plot_index = pd.to_datetime(ic_rolling.index.astype(int).astype(str), format='%Y%m%d')

    fig, ax = plt.subplots(figsize=(11, 4))
    for feat in top_feats:
        ax.plot(plot_index, ic_rolling[feat], label=feat, alpha=0.85)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_xlabel('Date (trading days, training period 2010–2013)')
    ax.set_ylabel(f'Rolling {window}-day mean IC')
    ax.set_title(f'Feature IC stability over time (rolling {window}-day window)\n'
                 'Persistent non-zero IC indicates a genuine, stationary signal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Walk-forward CV bar chart
# ---------------------------------------------------------------------------

def walk_forward_bar_chart(wf_results, save_path=None):
    """Grouped bar chart of val R² stability across walk-forward folds.

    Args:
        wf_results: DataFrame with columns model, val_year, val_r2.
        save_path:  If given, save figure to this path.
    """
    model_names = wf_results['model'].unique()
    val_years   = sorted(wf_results['val_year'].unique())
    n_models = len(model_names)
    x = np.arange(len(val_years))
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        subset = wf_results[wf_results['model'] == mname].sort_values('val_year')
        r2_vals = subset['val_r2'].values
        offsets = x + (i - n_models / 2 + 0.5) * width
        ax.bar(offsets, r2_vals, width=width * 0.9, label=mname, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Val {y}' for y in val_years])
    ax.set_xlabel('Validation year')
    ax.set_ylabel('Weighted R² (√MDV_63)')
    ax.set_title('Walk-forward CV: model R² stability across validation years')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Permutation importance (MDA)
# ---------------------------------------------------------------------------

def run_permutation_importance(model, X, y, w, feat_names, title, save_path=None):
    """Compute and plot permutation (MDA) feature importance.

    Args:
        model:      Fitted sklearn-compatible model.
        X:          Feature matrix (2-D array).
        y:          Target values.
        w:          Sample weights.
        feat_names: List of feature names aligned with X columns.
        title:      Plot title.
        save_path:  If given, save figure to this path.

    Returns:
        DataFrame with columns Feature, MDA_importance, MDA_std (sorted descending).
    """
    perm = permutation_importance(
        model, X, y, n_repeats=8, random_state=42,
        sample_weight=w, scoring='r2', n_jobs=-1,
    )
    imp = pd.DataFrame({
        'Feature':        feat_names,
        'MDA_importance': perm.importances_mean,
        'MDA_std':        perm.importances_std,
    }).sort_values('MDA_importance', ascending=False)
    print(f'\n{title}')
    display(imp)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp['Feature'][::-1], imp['MDA_importance'][::-1],
            xerr=imp['MDA_std'][::-1], color='steelblue', alpha=0.85)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Mean decrease in weighted R²')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return imp


# ---------------------------------------------------------------------------
# Feature bin plots (top features vs target)
# ---------------------------------------------------------------------------

def feature_bin_plot(val_df, val_preds_arr, top_features, feature_cols, label,
                     n_bins=5, save_path=None):
    """Bin plot: actual vs predicted target across quintiles of each top feature.

    Args:
        val_df:        Validation DataFrame with feature columns and Target_model.
        val_preds_arr: 1-D array of model predictions aligned with val_df.
        top_features:  List of feature names to plot (up to 6).
        feature_cols:  Full list of feature column names.
        label:         Title suffix string.
        n_bins:        Number of quantile bins (default 5).
        save_path:     If given, save figure to this path.
    """
    val_plot = val_df[feature_cols + ['Target_model']].copy()
    val_plot['Pred_model'] = val_preds_arr
    n_top = min(6, len(top_features))
    top = top_features[:n_top]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for idx, feat in enumerate(top):
        ax = axes[idx]
        val_plot['_bin'] = pd.qcut(val_plot[feat], q=n_bins, labels=False, duplicates='drop')
        binned = val_plot.groupby('_bin', observed=True).agg(
            mean_actual=('Target_model', 'mean'),
            mean_pred=('Pred_model', 'mean'),
            count=('Target_model', 'count'),
        ).reset_index()
        ax.bar(binned['_bin'], binned['mean_actual'], alpha=0.7, label='Mean actual')
        ax.plot(binned['_bin'], binned['mean_pred'], 'o-', color='C1', label='Mean pred')
        ax.set_title(feat, fontsize=9)
        ax.axhline(0, color='gray', lw=0.7, ls=':')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    for idx in range(n_top, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f'Bin plots: top {n_top} features vs target — {label}', fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# White-paper prediction bin plot
# ---------------------------------------------------------------------------

def wp_bin_plot(val_df, val_preds_arr, label, n_buckets=20, save_path=None):
    """Prediction bin plot (n_buckets quantiles of Pred vs mean actual target).

    Args:
        val_df:        Validation DataFrame with columns Date and Target.
        val_preds_arr: 1-D array of predictions aligned with val_df.
        label:         Title suffix.
        n_buckets:     Number of quantile buckets (default 20).
        save_path:     If given, save figure to this path.

    Returns:
        DataFrame ``vp`` with columns Date, Pred, y_actual_plot, Pred_bin.
    """
    labels_wp = ['Q' + f'{i+1}' for i in range(n_buckets)]
    vp = val_df[['Date']].copy()
    vp['Pred']          = val_preds_arr
    vp['y_actual_plot'] = val_df['Target'].values
    vp['Pred_bin'] = pd.qcut(vp['Pred'], q=n_buckets, labels=labels_wp, duplicates='drop')
    bin_stats = vp.groupby('Pred_bin', observed=True)['y_actual_plot'].agg(['mean', 'std', 'count'])
    bin_stats['se'] = bin_stats['std'] / np.sqrt(bin_stats['count'])

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = range(len(bin_stats))
    ax.bar(x_pos, bin_stats['mean'], yerr=bin_stats['se'], capsize=3,
           color='steelblue', alpha=0.85, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats.index, rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Mean actual target')
    ax.set_title(f'Prediction bin plot ({n_buckets} buckets) — {label}')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return vp


# ---------------------------------------------------------------------------
# Rolling cross-sectional correlation plot
# ---------------------------------------------------------------------------

def rolling_corr_plot(vp, y_val, label, window=30, save_path=None):
    """30-day MA of cross-sectional correlation between predictions and target.

    Args:
        vp:        DataFrame with columns Date and Pred (output of wp_bin_plot).
        y_val:     1-D array of target values aligned with vp.
        label:     Title suffix.
        window:    Rolling window in trading days (default 30).
        save_path: If given, save figure to this path.
    """
    vp = vp.copy()
    vp['y_fit'] = y_val
    daily_corr = (
        vp.groupby('Date')
        .apply(lambda g: g['Pred'].corr(g['y_fit']))
        .reset_index(name='corr')
    )
    daily_corr['corr_ma'] = daily_corr['corr'].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_corr.index, daily_corr['corr'],    alpha=0.4, label='Daily cross-sectional corr')
    ax.plot(daily_corr.index, daily_corr['corr_ma'], color='C1', lw=2, label=f'{window}-day MA')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Trading days (2014)')
    ax.set_ylabel('Cross-sectional correlation')
    ax.set_title(f'{window}-day MA of cross-sectional correlation — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Drift plot
# ---------------------------------------------------------------------------

def drift_plot(vp, label, save_path=None):
    """Prediction bin drift over time (quartiles of Pred vs mean actual per date).

    Args:
        vp:        DataFrame with columns Date, Pred, y_actual_plot.
        label:     Title suffix.
        save_path: If given, save figure to this path.
    """
    vd = vp.copy()
    vd['Bin'] = pd.qcut(vd['Pred'], q=4, labels=False, duplicates='drop')
    drift_agg = vd.groupby(['Date', 'Bin'])['y_actual_plot'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    drift_agg['se'] = drift_agg['std'] / np.sqrt(drift_agg['count'].clip(lower=1))
    drift_by_date = drift_agg.pivot(index='Date', columns='Bin', values='mean')
    se_by_date    = drift_agg.pivot(index='Date', columns='Bin', values='se')

    fig, ax = plt.subplots(figsize=(12, 5))
    for b in sorted(drift_by_date.columns):
        m  = drift_by_date[b].values
        se = se_by_date[b].fillna(0).values
        x  = np.arange(len(m))
        ax.plot(x, m, label=f'Bin {int(b)+1}', lw=1.5)
        ax.fill_between(x, m - se, m + se, alpha=0.15)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_xlabel('Trading days (2014 validation)')
    ax.set_ylabel('Mean actual target')
    ax.set_title(f'Prediction bin drift over time — {label}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Bias analysis
# ---------------------------------------------------------------------------

def bias_plot(val_df, preds_arr, label, n_bkts=50, save_path=None):
    """Mean prediction per volatility and liquidity bucket.

    Args:
        val_df:    Validation DataFrame with EST_VOL_prev and MDV_63_prev columns.
        preds_arr: 1-D array of predictions aligned with val_df.
        label:     Title suffix.
        n_bkts:    Number of equal-frequency buckets (default 50).
        save_path: If given, save figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    overall_mean = np.mean(preds_arr)
    for ax, char, lbl in [
        (axes[0], 'EST_VOL_prev', 'Volatility (EST_VOL_prev) decile'),
        (axes[1], 'MDV_63_prev',  'Liquidity (MDV_63_prev) decile'),
    ]:
        bucket_stats = bucket_mean_prediction(val_df, preds_arr, char, n_buckets=n_bkts)
        ax.bar(range(len(bucket_stats)), bucket_stats['mean_pred'] - overall_mean,
               color='steelblue', alpha=0.85)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_xlabel(lbl)
        ax.set_ylabel('Mean prediction − overall mean')
        ax.set_title(f'{char} bias — {label}')
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def bias_spearman(val_df, preds_arr, label):
    """Print Spearman rank correlation of predictions against vol/liquidity characteristics.

    Args:
        val_df:    Validation DataFrame with EST_VOL_prev and MDV_63_prev columns.
        preds_arr: 1-D array of predictions aligned with val_df.
        label:     Display label.
    """
    bias_corr = spearman_vs_characteristics(
        preds_arr,
        {
            'EST_VOL_prev (volatility)': val_df['EST_VOL_prev'].values,
            'MDV_63_prev (liquidity)':   val_df['MDV_63_prev'].values,
        },
    )
    print(f'\nSpearman rank correlation — {label}:')
    display(bias_corr.round(4))


# ---------------------------------------------------------------------------
# Intraday drift helpers (for per-year appendix)
# ---------------------------------------------------------------------------

# Ordered 15-min timestamps per trading day (09:45 to 16:00)
_DAY_TIMES = [
    '09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45',
    '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00',
    '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00',
]  # 26 timestamps
_TIME_TO_IDX = {t: i for i, t in enumerate(_DAY_TIMES)}
_IDX_1530 = _TIME_TO_IDX['15:30']  # = 23


def _load_intraday_file(date_str, intraday_dir=INTRADAY_DIR):
    """Load and return intraday CSV for date_str (YYYYMMDD). Returns None if missing."""
    path = os.path.join(intraday_dir, f'{date_str}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={'Time': str})
    df['TimeStr'] = df['Time'].str[:5]
    return df


def build_drift_data(pred_df, date_strings, intraday_dir=INTRADAY_DIR,
                     n_bins=4, sample_frac=0.3, random_state=42):
    """Build intraday return signature data for drift plots.

    For each (Date, Id) in pred_df, collects CumReturnResid across the window
    [D-1 full day] + [D] + [D+1 full day], normalised so that
    CumReturnResid at 15:30 on D = 0.

    Args:
        pred_df:      DataFrame with columns Date, Id, Pred.
        date_strings: Sorted list of available date strings (YYYYMMDD).
        intraday_dir: Directory with intraday CSV files.
        n_bins:       Number of prediction quantile bins (default 4).
        sample_frac:  Fraction of dates to sample (default 0.3).
        random_state: RNG seed.

    Returns:
        DataFrame with columns Bin, t_idx, mean_ret, se_ret.
    """
    rng = np.random.default_rng(random_state)
    dates = pred_df['Date'].unique()
    n_sample = max(1, int(len(dates) * sample_frac))
    sampled_dates = rng.choice(dates, size=n_sample, replace=False)
    sub = pred_df[pred_df['Date'].isin(sampled_dates)].copy()

    ds_list = sorted(date_strings)
    ds_idx  = {d: i for i, d in enumerate(ds_list)}

    sub['Bin'] = pd.qcut(sub['Pred'], q=n_bins, labels=False, duplicates='drop')

    all_chunks = []

    for date_int in sampled_dates:
        date_str = str(int(date_int))
        if date_str not in ds_idx:
            continue
        pos = ds_idx[date_str]

        d_prev = ds_list[pos - 1] if pos > 0 else None
        d_next = ds_list[pos + 1] if pos < len(ds_list) - 1 else None

        day_d    = _load_intraday_file(date_str, intraday_dir)
        day_prev = _load_intraday_file(d_prev, intraday_dir) if d_prev else None
        day_next = _load_intraday_file(d_next, intraday_dir) if d_next else None

        if day_d is None:
            continue

        ref_d = (
            day_d[day_d['TimeStr'] == '15:30'][['Id', 'CumReturnResid']]
            .set_index('Id')['CumReturnResid']
        )

        date_preds = sub[sub['Date'] == date_int].set_index('Id')['Bin']
        common_ids = set(ref_d.index) & set(date_preds.index)
        if not common_ids:
            continue

        def _collect(day_df, day_offset_base):
            if day_df is None:
                return None
            mask = day_df['Id'].isin(common_ids) & day_df['TimeStr'].isin(_TIME_TO_IDX)
            s = day_df[mask].copy()
            if s.empty:
                return None
            s['t_rel'] = day_offset_base + s['TimeStr'].map(_TIME_TO_IDX) - _IDX_1530
            s['delta'] = s['CumReturnResid'] - s['Id'].map(ref_d)
            s['Bin']   = s['Id'].map(date_preds)
            return s[['Bin', 't_rel', 'delta']].dropna()

        for day_df, offset in [(day_prev, -26), (day_d, 0), (day_next, 26)]:
            chunk = _collect(day_df, offset)
            if chunk is not None and not chunk.empty:
                all_chunks.append(chunk)

    if not all_chunks:
        return pd.DataFrame(columns=['Bin', 't_idx', 'mean_ret', 'se_ret'])

    raw = pd.concat(all_chunks, ignore_index=True)
    raw.columns = ['Bin', 't_idx', 'delta']
    agg = raw.groupby(['Bin', 't_idx'])['delta'].agg(['mean', 'std', 'count']).reset_index()
    agg['se'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
    return agg.rename(columns={'mean': 'mean_ret', 'se': 'se_ret'})
