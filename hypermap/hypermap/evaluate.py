"""
hypermap/evaluate.py

Evaluation metrics and visualisation for HyperMAP and comparable models.

All model results must first be converted to the unified format using
prepare_result_dict() before being passed to any evaluation function.

Unified result format
---------------------
    {
        'gene_list' : list of str,
        'donor1'    : { pert : { 'pred_delta': np.ndarray,
                                 'true_delta': np.ndarray } },
        ...
    }

Functions
---------
    prepare_result_dict     : convert any raw model dict to unified format
    plot_metric_boxgrid     : 1x3 panel, 3 metrics, significance vs base, median table
    plot_context_boxplot    : one subplot per context, Pearson correlation
    top_genes_global        : top-k genes for a pert from adata
    plot_top20_corr_for_gene: scatter plots pred vs true for a specific pert

Usage
-----
    from hypermap.evaluate import (
        prepare_result_dict,
        plot_metric_boxgrid,
        plot_context_boxplot,
        top_genes_global,
        plot_top20_corr_for_gene,
    )
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 0. prepare_result_dict
# ─────────────────────────────────────────────────────────────────────────────

def prepare_result_dict(obj, obj_type: str, var_names) -> Dict:
    """
    Normalizes any raw model result dict into the unified format.

    Unified format
    --------------
        'gene_list' : list of str
        donor       : { pert : { 'pred_delta': np.ndarray,
                                 'true_delta': np.ndarray } }

    Raw formats accepted
    --------------------
        'loaded_object' : { donor -> { pert -> { pred_delta, true_delta } } }
        'gears'         : { donor -> { pert+ctrl -> { pred_mean, true_mean } } }
        'scpram'        : { donor -> { pert -> { predicted, truth } } }
        'scgpt'         : { d1 -> { donor -> { pert+ctrl -> { pred_delta,
                                                              true_delta } } } }

    Parameters
    ----------
    obj      : raw result dict in any of the formats above
    obj_type : 'loaded_object' | 'gears' | 'scpram' | 'scgpt'
    var_names: adata.var_names for this model (Index or list)

    Returns
    -------
    dict with 'gene_list' key + donor keys in unified format
    """
    unified = {'gene_list': list(var_names)}

    if obj_type == 'loaded_object':
        for donor, perts in obj.items():
            if perts is None:
                continue
            unified[donor] = {}
            for pert, s in perts.items():
                if s is None or s.get('true_delta') is None:
                    continue
                unified[donor][pert] = {
                    'pred_delta': np.asarray(s['pred_delta']).ravel(),
                    'true_delta': np.asarray(s['true_delta']).ravel(),
                }

    elif obj_type == 'scpram':
        for donor, perts in obj.items():
            unified[donor] = {}
            for pert, s in perts.items():
                unified[donor][pert] = {
                    'pred_delta': np.asarray(s['predicted']).ravel(),
                    'true_delta': np.asarray(s['truth']).ravel(),
                }

    elif obj_type == 'gears':
        for donor, perts in obj.items():
            unified[donor] = {}
            for pert, s in perts.items():
                unified[donor][pert.replace('+ctrl', '')] = {
                    'pred_delta': np.asarray(s['pred_mean']).ravel(),
                    'true_delta': np.asarray(s['true_mean']).ravel(),
                }

    elif obj_type == 'scgpt':
        for d1, contexts in obj.items():
            for donor, perts in contexts.items():
                if donor not in unified:
                    unified[donor] = {}
                for pert, s in perts.items():
                    unified[donor][pert.replace('+ctrl', '')] = {
                        'pred_delta': np.asarray(s['pred_delta']).ravel(),
                        'true_delta': np.asarray(s['true_delta']).ravel(),
                    }

    else:
        raise ValueError(
            f"Unknown obj_type '{obj_type}'. "
            "Must be one of: 'loaded_object', 'gears', 'scpram', 'scgpt'."
        )

    return unified


# ─────────────────────────────────────────────────────────────────────────────
# Internal metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_idx(true: np.ndarray, k: int = 20) -> np.ndarray:
    """Top-k gene indices by absolute true delta."""
    t = min(k, len(true))
    return np.argpartition(np.abs(true), -t)[-t:]


def _pearson(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if len(a) < 2:
        return np.nan, np.nan
    r, p = pearsonr(a, b)
    return float(r), float(p)


def _topk_recall(pred: np.ndarray, true: np.ndarray, k: int = 20) -> float:
    """
    Fraction of true top-k genes (by abs delta) that appear in predicted top-k.
    Range [0, 1]. No threshold needed.
    """
    k       = min(k, len(true))
    true_top = set(np.argpartition(np.abs(true), -k)[-k:])
    pred_top = set(np.argpartition(np.abs(pred), -k)[-k:])
    return float(len(true_top & pred_top) / k)


def _stars(p: float) -> str:
    if np.isnan(p):  return ''
    if p < 1e-4:     return '****'
    if p < 1e-3:     return '***'
    if p < 1e-2:     return '**'
    if p < 0.05:     return '*'
    return 'ns'


def _sig_test(a: np.ndarray, b: np.ndarray, test: str = 't-test_ind') -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if test == 'mann-whitney':
        _, p = mannwhitneyu(a, b, alternative='two-sided')
    else:
        _, p = ttest_ind(a, b, equal_var=False)
    return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Unified result iterator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_results(obj: Dict):
    """
    Yields (donor, pert, pred_array, true_array) from a unified result dict.
    Skips the 'gene_list' key automatically.
    """
    for donor, perts in obj.items():
        if donor == 'gene_list':
            continue
        if perts is None:
            continue
        for pert, s in perts.items():
            if s is None or s.get('true_delta') is None:
                continue
            yield (donor, pert,
                   np.asarray(s['pred_delta']).ravel(),
                   np.asarray(s['true_delta']).ravel())


# ─────────────────────────────────────────────────────────────────────────────
# Gene index resolution helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_idx(
    donor:          str,
    pert:           str,
    true:           np.ndarray,
    gene_list:      List[str],
    top_genes_dict: Optional[Dict],
    k:              int,
) -> np.ndarray:
    """
    Returns indices into pred/true arrays for top-k genes.

    If top_genes_dict is provided and contains an entry for (donor, pert),
    resolves gene names to indices via gene_list.
    Falls back to abs-delta selection otherwise.

    Any gene name in top_genes_dict not found in gene_list is silently skipped.
    """
    if (top_genes_dict is not None
            and donor in top_genes_dict
            and pert  in top_genes_dict[donor]):
        genes = top_genes_dict[donor][pert]
        idx   = [gene_list.index(g) for g in genes if g in gene_list]
        if len(idx) == 0:
            # no supplied genes found in this model's gene list — fall back
            return _get_idx(true, k)
        return np.array(idx)
    return _get_idx(true, k)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics averaged across donors
# ─────────────────────────────────────────────────────────────────────────────

def get_donor_averaged_metrics(
    obj:            Dict,
    k:              int            = 20,
    top_genes_dict: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-perturbation metrics averaged across all donors.

    Metrics
    -------
        Top-k Pearson : Pearson r on top-k genes
        Top-k Recall  : fraction of true top-k genes in predicted top-k
        Top-k MSE     : MSE on top-k genes

    Parameters
    ----------
    obj            : unified result dict (from prepare_result_dict)
    k              : number of top genes. Default 20
    top_genes_dict : optional donor -> pert -> [gene_names] to override
                     abs-delta gene selection. Falls back to abs-delta if
                     a donor/pert key is missing.

    Returns
    -------
    { pert_name -> { metric_name -> mean_value } }
    """
    gene_list  = obj.get('gene_list', [])
    collector: Dict[str, list] = {}

    for donor, pert, pred, true in _iter_results(obj):
        mask = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[mask], true[mask]
        if pred.size < 2:
            continue

        idx    = _resolve_idx(donor, pert, true, gene_list, top_genes_dict, k)
        pred_k = pred[idx]
        true_k = true[idx]

        r_top, p_top = _pearson(pred_k, true_k)
        collector.setdefault(pert, []).append({
            f'Top-{k} Pearson':   r_top,
            f'Top-{k} Pearson p': p_top,
            f'Top-{k} Recall':    _topk_recall(pred, true, k),
            f'Top-{k} MSE':       float(np.mean((pred_k - true_k) ** 2)),
        })

    return {
        pert: {
            metric: float(np.nanmean([v[metric] for v in vals]))
            for metric in vals[0]
        }
        for pert, vals in collector.items()
    }


def build_metric_df(
    objs_dict:      Dict,
    k:              int            = 20,
    top_genes_dict: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Long-form DataFrame from multiple unified model result dicts.
    Columns: model, pert, metric, value

    Parameters
    ----------
    objs_dict      : { model_name: unified_result_dict }
    k              : number of top genes. Default 20
    top_genes_dict : optional donor -> pert -> [gene_names]
    """
    rows = []
    for model, obj in objs_dict.items():
        perts = get_donor_averaged_metrics(obj, k, top_genes_dict)
        for pert, metrics in perts.items():
            for metric, val in metrics.items():
                rows.append({'model': model, 'pert': pert,
                             'metric': metric, 'value': val})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Shared style helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':       10,
        'axes.titlesize':  11,
        'axes.labelsize':  10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })
    sns.set_style('white', {
        'axes.grid':      True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.color':     '#CCCCCC',
        'axes.edgecolor': 'grey',
        'axes.linewidth': 1.0,
    })


def _header(ax, title: str):
    ax.add_patch(Rectangle(
        (0, 1.0), 1, 0.15,
        transform=ax.transAxes,
        facecolor='#E6E6E6', edgecolor='grey',
        linewidth=1.0, clip_on=False,
    ))
    ax.text(0.5, 1.07, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=11,
            fontweight='bold', clip_on=False)


def _add_stars(ax, dsub, models, base_model, test):
    """Significance stars above each non-base model box."""
    base_vals = dsub[dsub['model'] == base_model]['value'].values
    for m_idx, model in enumerate(models):
        if model == base_model:
            continue
        model_vals = dsub[dsub['model'] == model]['value'].values
        p    = _sig_test(base_vals, model_vals, test)
        star = _stars(p)
        if star:
            max_val = np.nanpercentile(model_vals, 95) if len(model_vals) > 0 else 0
            ax.text(m_idx, max_val + 0.01, star,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')


def _save(fig, savepath: Optional[str], dpi: int = 300):
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
        print(f"[HyperMAP] Figure saved → {savepath}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. plot_metric_boxgrid
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_boxgrid(
    objs_dict:      Dict,
    colors_dict:    Dict[str, str],
    models:         List[str],
    base_model:     str            = 'HyperMap',
    figsize:        Tuple          = (10, 4),
    k:              int            = 20,
    sig_test:       str            = 't-test_ind',
    top_genes_dict: Optional[Dict] = None,
    savepath:       Optional[str]  = None,
    dpi:            int            = 300,
):
    """
    1×3 panel comparing models on Top-k Pearson, Top-k Recall, Top-k MSE.

    Significance stars shown vs base_model above each non-base box.
    Median summary table printed below the figure.

    Parameters
    ----------
    objs_dict      : { model_name: unified_result_dict }
    colors_dict    : { model_name: hex_color }
    models         : ordered list of model names
    base_model     : reference model for significance tests. Default 'HyperMap'
    figsize        : figure size. Default (10, 4)
    k              : top-k genes. Default 20
    sig_test       : 't-test_ind' or 'mann-whitney'. Default 't-test_ind'
    top_genes_dict : optional donor -> pert -> [gene_names] to override
                     abs-delta gene selection
    savepath       : optional path to save as SVG/PNG
    dpi            : resolution. Default 300

    Returns
    -------
    fig : matplotlib Figure
    df  : long-form pd.DataFrame
    """
    _apply_style()
    df = build_metric_df(objs_dict, k, top_genes_dict)

    METRICS = [f'Top-{k} Pearson', f'Top-{k} Recall', f'Top-{k} MSE']
    YLIMS   = {
        f'Top-{k} Pearson': (-1.05, 1.15),
        f'Top-{k} Recall':  (-0.05, 1.15),
        f'Top-{k} MSE':     None,
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, metric in enumerate(METRICS):
        ax   = axes[i]
        dsub = df[df['metric'] == metric]

        sns.boxplot(
            data=dsub, x='model', y='value',
            palette=colors_dict, order=models, ax=ax,
            showfliers=False, width=0.7, linewidth=1.3,
            boxprops=dict(alpha=0.30),
        )
        sns.stripplot(
            data=dsub, x='model', y='value',
            palette=colors_dict, order=models, ax=ax,
            size=2, jitter=True, alpha=0.45,
        )

        _header(ax, metric)
        _add_stars(ax, dsub, models, base_model, sig_test)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=35, length=2, width=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)

        if YLIMS[metric] is not None:
            ax.set_ylim(YLIMS[metric])
        else:
            hi = dsub['value'].quantile(0.99)
            ax.set_ylim(0, hi * 1.15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors_dict[m], alpha=0.8)
               for m in models]
    fig.legend(handles, models, loc='upper center',
               bbox_to_anchor=(0.5, 1.05), frameon=False,
               ncol=len(models), fontsize=10)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    _save(fig, savepath, dpi)

    # ── Median summary table ──────────────────────────────────────────────────
    print("\n── Median summary ──────────────────────────────────────────────")
    summary_rows = []
    for model in models:
        mdf = df[df['model'] == model]
        row = {'Model': model}
        for metric in METRICS:
            vals = mdf[mdf['metric'] == metric]['value'].dropna()
            row[f'Median {metric}'] = (
                round(float(np.median(vals)), 4) if len(vals) else np.nan
            )
        pvals = mdf[mdf['metric'] == f'Top-{k} Pearson p']['value'].dropna()
        row[f'Mean Top-{k} Pearson p'] = (
            round(float(np.median(pvals)), 4) if len(pvals) else np.nan
        )
        if model != base_model:
            for metric in METRICS:
                base_vals  = df[(df['model'] == base_model) &
                                (df['metric'] == metric)]['value'].dropna().values
                model_vals = mdf[mdf['metric'] == metric]['value'].dropna().values
                p = _sig_test(base_vals, model_vals, sig_test)
                row[f'p ({metric} vs {base_model})'] = (
                    round(p, 4) if not np.isnan(p) else np.nan
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index('Model')
    print(summary_df.to_string())
    print("────────────────────────────────────────────────────────────────\n")

    return fig, df


# ─────────────────────────────────────────────────────────────────────────────
# 2. plot_context_boxplot
# ─────────────────────────────────────────────────────────────────────────────

def plot_context_boxplot(
    objs_dict:      Dict,
    colors_dict:    Dict[str, str],
    models:         List[str],
    base_model:     str            = 'HyperMap',
    top_k:          bool           = True,
    k:              int            = 20,
    n_cols:         int            = 5,
    figsize:        Optional[Tuple] = None,
    sig_test:       str            = 't-test_ind',
    top_genes_dict: Optional[Dict] = None,
    savepath:       Optional[str]  = None,
    dpi:            int            = 300,
):
    """
    One subplot per donor/context comparing models on Pearson correlation.

    Parameters
    ----------
    objs_dict      : { model_name: unified_result_dict }
    colors_dict    : { model_name: hex_color }
    models         : ordered list of model names
    base_model     : reference for significance stars. Default 'HyperMap'
    top_k          : True = top-k genes, False = all genes. Default True
    k              : number of top genes when top_k=True. Default 20
    n_cols         : subplots per row. Default 5
    figsize        : optional figure size (auto if None)
    sig_test       : significance test. Default 't-test_ind'
    top_genes_dict : optional donor -> pert -> [gene_names] to override
                     abs-delta gene selection
    savepath       : optional save path
    dpi            : resolution. Default 300

    Returns
    -------
    fig : matplotlib Figure
    df  : long-form pd.DataFrame  columns: model, context_cell, pert, value
    """
    _apply_style()

    rows = []
    for model, obj in objs_dict.items():
        gene_list = obj.get('gene_list', [])
        for donor, pert, pred, true in _iter_results(obj):
            mask = np.isfinite(pred) & np.isfinite(true)
            pred, true = pred[mask], true[mask]
            if pred.size < 2:
                continue
            if top_k:
                idx  = _resolve_idx(donor, pert, true,
                                    gene_list, top_genes_dict, k)
                pred = pred[idx]
                true = true[idx]
            r, _ = _pearson(pred, true)
            rows.append({'model': model, 'context_cell': donor,
                         'pert': pert, 'value': r})

    df       = pd.DataFrame(rows)
    contexts = df['context_cell'].unique()
    n_ctx    = len(contexts)
    n_rows   = (n_ctx + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 2.5, n_rows * 3.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=False, sharey=True)
    axes = np.array(axes).flatten()

    metric_label = f'Top-{k} Pearson' if top_k else 'Pearson'

    for i, ctx in enumerate(contexts):
        ax   = axes[i]
        dsub = df[df['context_cell'] == ctx]

        sns.boxplot(
            data=dsub, x='model', y='value',
            palette=colors_dict, order=models, ax=ax,
            showfliers=False, width=0.7, linewidth=1.3,
            boxprops=dict(alpha=0.30),
        )
        sns.stripplot(
            data=dsub, x='model', y='value',
            palette=colors_dict, order=models, ax=ax,
            size=2, jitter=True, alpha=0.45,
        )

        _header(ax, ctx)
        _add_stars(ax, dsub, models, base_model, sig_test)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(-1.05, 1.15)
        ax.tick_params(axis='x', rotation=40, length=2, width=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors_dict[m], alpha=0.8)
               for m in models]
    fig.legend(handles, models, loc='upper center',
               bbox_to_anchor=(0.5, 0.01), frameon=False,
               ncol=min(len(models), 4), fontsize=10)

    fig.supxlabel('Models', fontsize=12, fontweight='bold', y=0.03)
    fig.supylabel(metric_label, fontsize=12, fontweight='bold', x=0.04)
    fig.suptitle('Perturbation prediction per context', fontsize=13, y=0.99)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    _save(fig, savepath, dpi)

    return fig, df


# ─────────────────────────────────────────────────────────────────────────────
# 3. top_genes_global
# ─────────────────────────────────────────────────────────────────────────────

def top_genes_global(
    adata,
    pert:     str,
    contexts: List[str],
    n_top:    int           = 20,
    layer:    Optional[str] = None,
) -> List[str]:
    """
    Top-n genes (by abs mean delta) for a perturbation across specified contexts.
    Computes delta as mean(pert) - mean(ctrl) per context, then averages.
    Excludes the perturbed gene itself from ranking.

    Parameters
    ----------
    adata    : AnnData with obs['context_cell'] and obs['condition']
    pert     : perturbation name matching adata.obs['condition']
    contexts : list of context_cell values to include
    n_top    : number of top genes to return. Default 20
    layer    : layer to use for expression. None = adata.X. Default None

    Returns
    -------
    list of str : gene names ordered by abs delta ascending
                  (same order as np.argsort, last = highest)
    """
    deltas = []
    for ctx in contexts:
        ctrl = adata[
            (adata.obs['context_cell'] == ctx) &
            (adata.obs['condition'] == 'ctrl')
        ]
        pert_data = adata[
            (adata.obs['context_cell'] == ctx) &
            (adata.obs['condition'] == pert)
        ]
        if ctrl.shape[0] == 0 or pert_data.shape[0] == 0:
            continue

        Xc = ctrl.X      if layer is None else ctrl.layers[layer]
        Xp = pert_data.X if layer is None else pert_data.layers[layer]

        mu_c = np.asarray(Xc.mean(axis=0)).ravel()
        mu_p = np.asarray(Xp.mean(axis=0)).ravel()
        deltas.append(mu_p - mu_c)

    if not deltas:
        raise ValueError(
            f"No data found for pert='{pert}' in contexts {contexts}.\n"
            "Check adata.obs['context_cell'] and adata.obs['condition']."
        )

    mean_delta = np.vstack(deltas).mean(axis=0)

    if pert in adata.var_names:
        pert_idx = list(adata.var_names).index(pert)
        mean_delta[pert_idx] = 0

    top_idx = np.argsort(np.abs(mean_delta))[-n_top:]
    return list(adata.var_names[top_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 4. plot_top20_corr_for_gene
# ─────────────────────────────────────────────────────────────────────────────

def plot_top20_corr_for_gene(
    loaded_object: Dict,
    top_20_donor:  List[str],
    gene:          str,
    donors:        List[str],
    figsize:       Tuple               = (10, 4),
    label_top:     bool                = True,
    label_genes:   Optional[List[str]] = None,
    savepath:      Optional[str]       = None,
    dpi:           int                 = 300,
):
    """
    Scatter plots of predicted vs true delta for a specific perturbation.
    One subplot per donor. Works with unified result dict.

    Parameters
    ----------
    loaded_object : unified result dict (from prepare_result_dict)
    top_20_donor  : gene names to plot (from top_genes_global or user-supplied)
    gene          : perturbation name
    donors        : donor/context names to plot
    figsize       : figure size. Default (10, 4)
    label_top     : label top-2 and bottom-2 genes by true value. Default True
    label_genes   : optional specific gene names to label instead
    savepath      : optional save path
    dpi           : resolution. Default 300

    Returns
    -------
    fig : matplotlib Figure
    """
    _apply_style()

    gene_list = loaded_object.get('gene_list', [])

    fig, axes = plt.subplots(1, len(donors), figsize=figsize)
    if len(donors) == 1:
        axes = [axes]

    for ax, donor in zip(axes, donors):

        # ── Guard: missing data ───────────────────────────────────────────
        if donor not in loaded_object or loaded_object[donor] is None:
            ax.set_title(f"{donor}\n(no data)", fontsize=11)
            ax.axis('off')
            continue
        if gene not in loaded_object[donor]:
            ax.set_title(f"{donor}\n(pert not found)", fontsize=11)
            ax.axis('off')
            continue
        s = loaded_object[donor][gene]
        if s is None or s.get('true_delta') is None:
            ax.set_title(f"{donor}\n(no ground truth)", fontsize=11)
            ax.axis('off')
            continue

        # ── Gene indices ──────────────────────────────────────────────────
        idx            = [gene_list.index(g) for g in top_20_donor
                          if g in gene_list]
        selected_genes = [gene_list[i] for i in idx]

        pred = np.asarray(s['pred_delta']).ravel()[idx]
        true = np.asarray(s['true_delta']).ravel()[idx]

        # ── Scatter ───────────────────────────────────────────────────────
        ax.scatter(pred, true, s=45, color='#84a98c', alpha=0.6)

        # ── Labels ────────────────────────────────────────────────────────
        if label_genes is not None:
            for i, g in enumerate(selected_genes):
                if g in label_genes:
                    ax.scatter(pred[i], true[i], s=55, color='#84a98c',
                               edgecolor='black', linewidth=0.8, zorder=3)
                    ax.text(pred[i], true[i], g, fontsize=9,
                            color='black', ha='left', va='bottom')
        elif label_top:
            order   = np.argsort(true)
            bottom2 = order[:2]
            top2    = order[-2:]
            for ii in bottom2:
                ax.scatter(pred[ii], true[ii], s=55, color='#84a98c',
                           edgecolor='black', linewidth=0.8, zorder=3)
                ax.text(pred[ii], true[ii], selected_genes[ii],
                        color='firebrick', fontsize=9, ha='right', va='top')
            for ii in top2:
                ax.scatter(pred[ii], true[ii], s=55, color='#84a98c',
                           edgecolor='black', linewidth=0.8, zorder=3)
                ax.text(pred[ii], true[ii], selected_genes[ii],
                        color='navy', fontsize=9, ha='left', va='bottom')

        # ── Pearson r ─────────────────────────────────────────────────────
        r, _ = _pearson(true, pred)
        ax.set_title(f"{donor}  (r = {r:.2f})", fontsize=12, fontweight='bold')

        ax.set_xlabel('Predicted Δ-expression')
        ax.set_ylabel('True Δ-expression')

        min_lim = min(true.min(), pred.min())
        max_lim = max(true.max(), pred.max())
        pad     = 0.05 * (max_lim - min_lim)
        ax.set_xlim(min_lim - pad, max_lim + pad)
        ax.set_ylim(min_lim - pad, max_lim + pad)
        ax.set_aspect('equal', adjustable='box')

        ax.axhline(0, linestyle='--', color='grey', linewidth=1)
        ax.axvline(0, linestyle='--', color='grey', linewidth=1)
        ax.plot([min_lim, max_lim], [min_lim, max_lim],
                '--', color='grey', linewidth=1)

    plt.suptitle(f'Perturbation: {gene}', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, savepath, dpi)

    return fig
    