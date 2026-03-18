"""
HyperMAP
========
Meta-learning model for single-cell perturbation prediction.

Quick start
-----------
    from hypermap import HyperMAP

    model = HyperMAP(
        adata        = adata,          # AnnData with obs['context_cell'] and obs['condition']
        gene_emb     = gene_emb,       # dict { pert_name: embedding_vector } 
        project_name = "my_project",   # cache directory name
    )

    # Leave-one-context-out evaluation
    loo_results = model.loo()

    # Train on some contexts, predict on others
    adata_pred = model.train_predict(
        predict_contexts    = ["cell_1", "cell_2"],
        use_gene_emb_perts  = True,
        output_format       = "pseudobulk",
    )

    # Train on all, impute all
    adata_imp = model.impute(output_format="singlecell", n_cells_per_pert=20)

Required adata format
---------------------
    adata.X                 dense numpy array  (call .toarray() if sparse)
    adata.obs['context_cell']   biological context identifier (cell line, donor, ...)
    adata.obs['condition']      perturbation name OR 'ctrl' for unperturbed cells

    Every context_cell must have at least one 'ctrl' row.

Adaptation gene selection
-------------------------
    selection_strategy = 'random'          # default, no prior knowledge needed
    selection_strategy = 'least_consistent'# perts most variable across training contexts
    selection_strategy = 'functional'      # perts you supply explicitly
    functional_perts   = ['TP53', 'MYC']  # required when strategy='functional'

    When to use each:
        random          → default, works well when test context is similar to training
        least_consistent→ test context is biologically diverse from training
        functional      → you have prior knowledge of informative perts for test context


---------------------
    model = HyperMAP(
        adata         = adata,
        gene_emb      = gene_emb,
        holdout_perts = ["TP53", "BRCA1"],   # excluded from training AND adaptation
    )
    # Holdouts are still predicted at inference time (no ground truth in training).
"""

from .trainer  import HyperMAP
from .utils    import sample_cells
from .evaluate import plot_metric_boxgrid, build_metric_df, get_donor_averaged_metrics

__all__ = [
    "HyperMAP",
    "sample_cells",
    "plot_metric_boxgrid",
    "build_metric_df",
    "get_donor_averaged_metrics",
]
__version__ = "0.1.0"