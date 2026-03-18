"""
hypermap/utils.py
Validation, embedding construction, and output formatting utilities.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import scipy.sparse


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_adata(adata, gene_emb: Optional[Dict] = None) -> None:
    """
    Full validation of input AnnData before any processing.
    Hard errors stop execution. Warnings continue with degraded behaviour.

    Required:
        - adata.X is a dense numpy array (NOT sparse)
        - adata.obs contains 'context_cell' column
        - adata.obs contains 'condition' column
        - Every unique context_cell value has at least one row where condition == 'ctrl'

    Warnings:
        - Any non-ctrl condition in adata not found in gene_emb → those perts dropped
        - If gene_emb is None → one-hot vectors will be used
    """

    # ── 1. Dense check ────────────────────────────────────────────────────────
    if scipy.sparse.issparse(adata.X):
        raise TypeError(
            "adata.X is a sparse matrix. HyperMAP requires a dense numpy array.\n"
            "Fix: run  adata.X = adata.X.toarray()  before passing to HyperMAP."
        )
    if not isinstance(adata.X, np.ndarray):
        raise TypeError(
            f"adata.X must be a numpy ndarray, got {type(adata.X)}.\n"
            "Fix: run  adata.X = np.array(adata.X)  before passing to HyperMAP."
        )

    # ── 2. Required columns ───────────────────────────────────────────────────
    if 'context_cell' not in adata.obs.columns:
        raise ValueError(
            "adata.obs must contain a column named 'context_cell'.\n"
            "This column identifies each biological context (cell line, donor, etc.).\n"
            "Fix: rename your context column →  adata.obs.rename("
            "columns={'your_col': 'context_cell'}, inplace=True)"
        )
    if 'condition' not in adata.obs.columns:
        raise ValueError(
            "adata.obs must contain a column named 'condition'.\n"
            "This column identifies each perturbation (gene name, drug, etc.) "
            "and must include 'ctrl' for unperturbed cells."
        )

    # ── 3. Every context must have ctrl cells ─────────────────────────────────
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    contexts_with_ctrl = set(adata.obs.loc[ctrl_mask, 'context_cell'].unique())
    all_contexts = set(adata.obs['context_cell'].unique())
    missing_ctrl = all_contexts - contexts_with_ctrl
    if missing_ctrl:
        raise ValueError(
            f"The following context_cell values have NO 'ctrl' cells:\n"
            f"  {sorted(missing_ctrl)}\n"
            "Every context must have at least one row with condition == 'ctrl'."
        )

    # ── 4. Embedding coverage warning ─────────────────────────────────────────
    if gene_emb is not None:
        adata_perts = set(adata.obs.loc[~ctrl_mask, 'condition'].unique())
        emb_perts   = set(gene_emb.keys()) - {'ctrl'}
        missing_in_emb = adata_perts - emb_perts
        if missing_in_emb:
            warnings.warn(
                f"{len(missing_in_emb)} perturbation(s) in adata have no embedding in gene_emb "
                f"and will be DROPPED from training/evaluation:\n"
                f"  {sorted(missing_in_emb)[:10]}{'...' if len(missing_in_emb) > 10 else ''}\n"
                "Supply embeddings for these conditions if you want them included.",
                UserWarning, stacklevel=3
            )
    else:
        warnings.warn(
            "No gene_emb supplied. HyperMAP will use one-hot vectors as perturbation embeddings.\n"
            "This is suitable for small perturbation sets but will not generalise to unseen perts.",
            UserWarning, stacklevel=3
        )


def validate_predict_contexts(adata, predict_contexts: List[str]) -> None:
    """Check all requested predict contexts exist in adata."""
    all_contexts = set(adata.obs['context_cell'].unique())
    missing = set(predict_contexts) - all_contexts
    if missing:
        raise ValueError(
            f"The following predict_contexts were not found in adata.obs['context_cell']:\n"
            f"  {sorted(missing)}"
        )
    remaining_train = all_contexts - set(predict_contexts)
    if len(remaining_train) == 0:
        raise ValueError(
            "No training contexts remain after removing predict_contexts.\n"
            "You must keep at least one context for training."
        )
    if len(remaining_train) < 4:
        warnings.warn(
            f"Only {len(remaining_train)} training context(s) available. "
            "HyperMAP performs best with more reference contexts (ideally ≥ 4). "
            "Meta-learning will use replacement sampling to fill meta-batches.",
            UserWarning, stacklevel=3
        )


def validate_holdout_perts(adata, holdout_perts: List[str]) -> None:
    """Check holdout perts exist in adata conditions."""
    all_perts = set(adata.obs.loc[adata.obs['condition'] != 'ctrl', 'condition'].unique())
    missing = set(holdout_perts) - all_perts
    if missing:
        warnings.warn(
            f"{len(missing)} holdout pert(s) not found in adata conditions and will be ignored:\n"
            f"  {sorted(missing)[:10]}",
            UserWarning, stacklevel=3
        )
    remaining = all_perts - set(holdout_perts)
    if len(remaining) == 0:
        raise ValueError(
            "All perturbations have been held out — no training signal would remain."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_cache_dir(project_name: str) -> str:
    """
    Create and return the cache directory path for a given project name.
    Warns if the directory already exists.
    """
    cache_dir = os.path.join("hypermap_cache", project_name)
    if os.path.exists(cache_dir):
        warnings.warn(
            f"Cache directory '{cache_dir}' already exists.\n"
            "Existing cached context data will be reused.\n"
            "If your data has changed, use a different project_name for a fresh run.",
            UserWarning, stacklevel=3
        )
    else:
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# ─────────────────────────────────────────────────────────────────────────────
# Embedding construction
# ─────────────────────────────────────────────────────────────────────────────

def build_one_hot_embeddings(conditions: List[str]) -> Dict[str, np.ndarray]:
    """
    Build one-hot embeddings for a list of perturbation condition names.
    The identity matrix row order matches the order of `conditions`.
    """
    n = len(conditions)
    one_hot = np.eye(n, dtype=np.float32)
    return {cond: one_hot[i] for i, cond in enumerate(conditions)}


def resolve_embeddings(
    adata,
    gene_emb: Optional[Dict],
    holdout_perts: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Return a single master embedding dict with consistent vectors.

    Rules:
    - gene_emb is always the source of truth.
    - If gene_emb is None, build one-hot from adata conditions.
    - Perts in adata but not in gene_emb are dropped (already warned in validate_adata).
    - holdout_perts are included in the returned dict (needed for inference),
      but callers are responsible for excluding them from training dataloaders.

    Returns:
        Dict mapping condition_name -> embedding vector (np.float32)
    """
    ctrl_mask = adata.obs['condition'] == 'ctrl'
    adata_perts = list(adata.obs.loc[~ctrl_mask, 'condition'].unique())

    if gene_emb is None:
        return build_one_hot_embeddings(adata_perts)

    # Filter to perts that exist in both adata and gene_emb
    valid_perts = [p for p in adata_perts if p in gene_emb]

    # Also include any gene_emb keys not in adata (for full inference)
    all_emb_perts = list(gene_emb.keys())
    all_perts = list(dict.fromkeys(valid_perts + all_emb_perts))  # preserve order, no dups

    result = {}
    for p in all_perts:
        if p == 'ctrl':
            continue
        if p in gene_emb:
            result[p] = np.array(gene_emb[p], dtype=np.float32)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_loo_results(
    results_dict: Dict,
    pert_encoder: Dict[str, int]
) -> Dict:
    """
    Results are already in the right shape from training loops.
    Just ensures the structure is:
        { context_name: { pert_name: { 'pred_delta': np.array, 'true_delta': np.array or None } } }
    """
    return results_dict


def build_pseudobulk_adata(predictions: Dict, var_names, source_label: str = 'predicted') -> 'sc.AnnData':
    """
    Build a pseudobulk AnnData from a nested dict:
        { context_name: { pert_name: { 'pred_delta': np.array } } }

    Each row = one (context, pert) combination.
    X = mean predicted delta.
    """
    import scanpy as sc

    rows_X   = []
    rows_obs = []

    for context, pert_dict in predictions.items():
        for pert, vals in pert_dict.items():
            pred = vals.get('pred_delta')
            if pred is None:
                continue
            # pred may be (n_cells, n_genes) or (n_genes,)
            pred = np.atleast_2d(pred)
            rows_X.append(pred.mean(axis=0))
            rows_obs.append({'context_cell': context, 'perturbation': pert})

    X   = np.vstack(rows_X).astype(np.float32)
    obs = pd.DataFrame(rows_obs)
    obs.index = [f"{r['context_cell']}_{r['perturbation']}" for _, r in obs.iterrows()]

    adata_out = sc.AnnData(
        X   = X,
        obs = obs,
        var = pd.DataFrame(index=var_names)
    )
    return adata_out


def build_singlecell_adata(predictions: Dict, var_names, n_cells: int = 20) -> 'sc.AnnData':
    """
    Build a single-cell style AnnData.
    Each (context, pert) contributes n_cells rows (sampled with replacement if needed).
    X = predicted delta per cell.
    """
    import scanpy as sc

    rows_X   = []
    rows_obs = []

    for context, pert_dict in predictions.items():
        for pert, vals in pert_dict.items():
            pred = vals.get('pred_delta')
            if pred is None:
                continue
            pred = np.atleast_2d(pred)
            # Sample or tile to exactly n_cells rows
            if pred.shape[0] >= n_cells:
                idx = np.random.choice(pred.shape[0], n_cells, replace=False)
            else:
                idx = np.random.choice(pred.shape[0], n_cells, replace=True)
            sampled = pred[idx]
            rows_X.append(sampled)
            for j in range(n_cells):
                rows_obs.append({
                    'context_cell':  context,
                    'perturbation':  pert,
                    'sample_idx':    j
                })

    X   = np.vstack(rows_X).astype(np.float32)
    obs = pd.DataFrame(rows_obs)
    obs.index = [
        f"{r['context_cell']}_{r['perturbation']}_{r['sample_idx']}"
        for _, r in obs.iterrows()
    ]

    adata_out = sc.AnnData(
        X   = X,
        obs = obs,
        var = pd.DataFrame(index=var_names)
    )
    return adata_out


# ─────────────────────────────────────────────────────────────────────────────
# Adaptation data guard
# ─────────────────────────────────────────────────────────────────────────────

def check_adaptation_data(n_samples: int, context: str) -> str:
    """
    Returns 'skip', 'thin', or 'ok' depending on available adaptation samples.
    Prints appropriate warnings.
    """
    if n_samples < 2:
        warnings.warn(
            f"[{context}] Only {n_samples} sample(s) available for adaptation. "
            "Skipping adaptation — using base meta-learned model directly.",
            UserWarning, stacklevel=2
        )
        return 'skip'
    if n_samples < 4:
        warnings.warn(
            f"[{context}] Only {n_samples} samples available for adaptation. "
            "Proceeding with a single batch, but results may be unreliable.",
            UserWarning, stacklevel=2
        )
        return 'thin'
    return 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# Data sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_cells(
    adata,
    condition_key:  str  = 'condition',
    context_key:    str  = 'context_cell',
    upper_limit:    int  = 400,
    keep_all_ctrl:  bool = True,
    equalize:       bool = False,
    random_seed:    int  = 42,
):
    """
    Sample cells from an AnnData object to balance conditions before model training.
    Call this before passing adata to HyperMAP if your data is imbalanced.

    Parameters
    ----------
    adata : AnnData
        Input data. Must have obs columns for condition and context.
    condition_key : str
        Column in adata.obs with perturbation labels. Default 'condition'.
    context_key : str
        Column in adata.obs with context labels. Default 'context_cell'.
    upper_limit : int
        Max cells per (condition, context) combination for non-ctrl conditions.
        Also used as target count when equalize=True. Default 400.
    keep_all_ctrl : bool
        If True, keep all ctrl cells regardless of count. Default True.
        Recommended — ctrl pool size affects matched pair quality in training.
    equalize : bool
        If False (default): cap conditions at upper_limit, keep smaller ones as-is.
        If True: bring ALL conditions to exactly upper_limit using replacement
                 sampling for small conditions. Warning: this duplicates cells
                 for rare conditions and can cause overfitting.
    random_seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    AnnData
        New AnnData with sampled cells. Original adata is not modified.

    Notes
    -----
    This function operates per (context, condition) pair — same condition in
    different contexts is treated independently. This preserves biological
    variability across contexts while controlling within-context imbalance.

    If you apply this BEFORE generating cache files, regenerate the cache
    using a new project_name to avoid loading stale cached data.
    """
    np.random.seed(random_seed)

    contexts   = adata.obs[context_key].unique()
    conditions = adata.obs[condition_key].unique()
    indices_to_keep = []

    for ctx in contexts:
        ctx_mask = adata.obs[context_key] == ctx

        for cond in conditions:
            combined_mask = ctx_mask & (adata.obs[condition_key] == cond)
            cell_indices  = np.where(combined_mask)[0]

            if len(cell_indices) == 0:
                continue

            if cond.lower() == 'ctrl' and keep_all_ctrl:
                indices_to_keep.extend(cell_indices)
                continue

            if equalize:
                replace = len(cell_indices) < upper_limit
                sampled = np.random.choice(
                    cell_indices, size=upper_limit, replace=replace
                )
                indices_to_keep.extend(sampled)
            else:
                if len(cell_indices) <= upper_limit:
                    indices_to_keep.extend(cell_indices)
                else:
                    sampled = np.random.choice(
                        cell_indices, size=upper_limit, replace=False
                    )
                    indices_to_keep.extend(sampled)

    sampled_adata = adata[indices_to_keep].copy()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("Cells per context and condition after sampling:")
    for ctx in contexts:
        print(f"\n  Context: {ctx}")
        ctx_mask = sampled_adata.obs[context_key] == ctx
        counts   = sampled_adata.obs[ctx_mask][condition_key].value_counts()
        print(counts.to_string())

    return sampled_adata