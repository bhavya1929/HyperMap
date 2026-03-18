"""
hypermap/trainer.py

Main HyperMAP class.
Three modes:
    - loo()            : leave-one-context-out cross-validation
    - train_predict()  : explicit train / predict split
    - impute()         : train on all, predict on all (or a subset)

Output format:
    - LOO              : dict  { context -> { pert -> { pred_delta, true_delta } } }
    - train_predict /
      impute           : AnnData  (pseudobulk OR single-cell)
"""

from __future__ import annotations

import copy
import gc
import os
import pickle
import warnings
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import higher

from .dataset  import PerturbDataset, collate_min2
from .model    import Net
from .utils    import (
    validate_adata,
    validate_predict_contexts,
    validate_holdout_perts,
    setup_cache_dir,
    resolve_embeddings,
    check_adaptation_data,
    build_pseudobulk_adata,
    build_singlecell_adata,
)


# ─────────────────────────────────────────────────────────────────────────────
# HyperMAP
# ─────────────────────────────────────────────────────────────────────────────

class HyperMAP:
    """
    HyperMAP: meta-learning model for single-cell perturbation prediction.

    Parameters
    ----------
    adata : AnnData
        Must contain:
            obs['context_cell']  – biological context identifier
            obs['condition']     – perturbation name or 'ctrl'
        adata.X must be a dense numpy array.

    gene_emb : dict, optional
        Mapping  { pert_name : embedding_vector (np.ndarray) }.
        If None, one-hot vectors are built automatically from adata conditions.
        gene_emb is always the single source of truth for embeddings —
        the same pert uses the same vector everywhere.

    project_name : str
        Used to name the cache directory (hypermap_cache/<project_name>/).
        Change this to force a fresh data cache.

    latent_dim : int
        Latent dimension for encoders. Default 64.

    hidden_dim : int
        Hidden dimension throughout the network. Default 256.

    training_epochs : int
        Number of meta-learning epochs. Default 50.

    batch_size : int
        Dataloader batch size. Default 512.

    meta_lr : float
        Outer (meta) optimizer learning rate. Default 0.0005.
        Rarely needs changing.

    inner_lr : float
        Inner loop (SGD) learning rate during meta-training. Default 0.005.
        Increase (0.008–0.01) when training contexts are biologically diverse.

    n_inner_steps : int
        Inner gradient steps per batch during meta-training. Default 5.

    n_adapt_genes : int
        Number of perturbations used during test-time adaptation. Default 10.
        Use 20 when the test context is biologically distant from training.

    n_adapt_steps : int
        Gradient steps per batch during test-time adaptation. Default 1.
        Increase to 3 when the test context is biologically distant from training.
        Tune together with inner_lr_adapt.

    inner_lr_adapt : float, optional
        Learning rate for test-time adaptation SGD. Defaults to inner_lr if not set.
        Lower this (e.g. half of inner_lr) if adaptation produces NaN or poor results
        for a specific context.

    grad_clip : float, optional
        Gradient norm clipping threshold applied to the meta gradient only.
        Default None (disabled). Set to 10.0 if meta-loss becomes NaN during training.
        Do not set below 5.0 as it will reduce performance.
        Inner loop gradients are not clipped.

    selection_strategy : str
        Strategy for selecting perturbations used during test-time adaptation.
        Default 'random' — first n_adapt_genes perts by integer ID order.
        Options:
            'random'          : default, no prior knowledge needed
            'least_consistent': selects perts with highest effect variability
                                across training contexts. Requires adata layers
                                to be computed (done automatically). Best when
                                your test context is biologically diverse from
                                training.
            'functional'      : use perts explicitly supplied via functional_perts.
                                Best when you have prior knowledge of which perts
                                are informative for your test context.

    functional_perts : list of str, optional
        Required when selection_strategy='functional'.
        List of condition names from adata.obs['condition'] to use for adaptation.
        Only perts present in the test context will be used — others are ignored.

    holdout_perts : list of str, optional
        Perturbations excluded from ALL training dataloaders.
        These are predicted at inference time only (no ground truth during training).
        Holdouts are also excluded from adaptation.

    seed : int
        Random seed. Default 1234.
    """

    def __init__(
        self,
        adata,
        gene_emb:         Optional[Dict]       = None,
        project_name:     str                  = "hypermap_project",
        latent_dim:       int                  = 64,
        hidden_dim:       int                  = 256,
        training_epochs:  int                  = 50,
        batch_size:       int                  = 512,
        meta_lr:          float                = 0.0005,
        inner_lr:         float                = 0.005,
        n_inner_steps:    int                  = 5,
        n_adapt_genes:    int                  = 10,
        n_adapt_steps:    int                  = 1,
        inner_lr_adapt:   Optional[float]      = None,
        grad_clip:        Optional[float]      = None,
        selection_strategy: str               = 'random',
        functional_perts: Optional[List[str]] = None,
        holdout_perts:    Optional[List[str]]  = None,
        seed:             int                  = 1234,
    ):
        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HyperMAP] Using device: {self.device}")

        # ── Seeds ─────────────────────────────────────────────────────────────
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ── Validate inputs ───────────────────────────────────────────────────
        validate_adata(adata, gene_emb)
        if holdout_perts:
            validate_holdout_perts(adata, holdout_perts)

        # ── Store config ──────────────────────────────────────────────────────
        self.adata           = adata.copy()
        self.project_name    = project_name
        self.latent_dim      = latent_dim
        self.hidden_dim      = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size      = batch_size
        self.meta_lr         = meta_lr
        self.inner_lr        = inner_lr
        self.n_inner_steps   = n_inner_steps
        self.n_adapt_genes   = n_adapt_genes
        self.n_adapt_steps   = n_adapt_steps
        self.inner_lr_adapt  = inner_lr_adapt if inner_lr_adapt is not None else inner_lr
        self.grad_clip       = grad_clip
        self.meta_batch_size = 4
        self.seed            = seed

        # ── Adaptation gene selection ──────────────────────────────────────────
        valid_strategies = {'random', 'least_consistent', 'functional'}
        if selection_strategy not in valid_strategies:
            raise ValueError(
                f"selection_strategy must be one of {valid_strategies}, "
                f"got '{selection_strategy}'"
            )
        if selection_strategy == 'functional' and not functional_perts:
            raise ValueError(
                "selection_strategy='functional' requires functional_perts to be provided.\n"
                "Pass a list of condition names from adata.obs['condition']."
            )
        self.selection_strategy = selection_strategy
        self.functional_perts   = set(functional_perts) if functional_perts else set()
        self._least_consistent_cache: Optional[Dict] = None  # computed lazily per fold

        # ── Holdout perts ─────────────────────────────────────────────────────
        self._holdout_names: set = set(holdout_perts) if holdout_perts else set()

        # ── Cache dir ─────────────────────────────────────────────────────────
        self.cache_dir = setup_cache_dir(project_name)

        # ── Embeddings (single source of truth) ───────────────────────────────
        self.gene_emb = resolve_embeddings(self.adata, gene_emb, holdout_perts)
        self.p_dim    = next(iter(self.gene_emb.values())).shape[0]
        self.x_dim    = self.adata.shape[1]

        # ── Perturbation encoder  (int IDs for dataloader bookkeeping) ────────
        # Built from adata conditions only (not from gene_emb extras).
        # Extras are handled at inference time via pert name → embedding lookup.
        all_conds = list(self.adata.obs['condition'].unique())
        self.pert_encoder: Dict[str, int] = {
            p: i for i, p in enumerate(all_conds)
        }

        # IDs of holdout perts (for fast filtering)
        self._holdout_ids: set = {
            self.pert_encoder[p]
            for p in self._holdout_names
            if p in self.pert_encoder
        }

        # ── Preprocess ────────────────────────────────────────────────────────
        self._preprocess_data()

        # ── Net placeholder (reset before each fold) ──────────────────────────
        self.Net: Optional[Net] = None
        self._context_dataloaders: Dict = {}

        print(
            f"[HyperMAP] Initialised. "
            f"Contexts: {len(self.adata.obs['context_cell'].unique())}  |  "
            f"Conditions: {len(self.pert_encoder)}  |  "
            f"Embedding dim: {self.p_dim}  |  "
            f"Genes: {self.x_dim}"
        )
        if self._holdout_names:
            print(f"[HyperMAP] Holdout perts: {len(self._holdout_names)}")


    # ─────────────────────────────────────────────────────────────────────────
    # Data preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess_data(self):
        """
        Compute context-specific control means, normalized expression (delta),
        and assign perturbation embeddings to every cell in adata.
        """
        print("[HyperMAP] Computing context-specific control means...")

        self._context_means: Dict[str, np.ndarray] = {}
        ctrl_mask = self.adata.obs['condition'] == 'ctrl'

        for ctx in self.adata.obs['context_cell'].unique():
            ctx_ctrl_mask = (self.adata.obs['context_cell'] == ctx) & ctrl_mask
            self._context_means[ctx] = np.mean(self.adata[ctx_ctrl_mask].X, axis=0)

        print("[HyperMAP] Computing delta (normalized expression)...")
        normalized_exp = np.zeros_like(self.adata.X)
        for ctx in self.adata.obs['context_cell'].unique():
            ctx_mask = self.adata.obs['context_cell'] == ctx
            normalized_exp[ctx_mask] = (
                self.adata[ctx_mask].X - self._context_means[ctx]
            )
        self.adata.layers['normalized_exp'] = normalized_exp

        print("[HyperMAP] Assigning perturbation embeddings...")
        emb_matrix = np.zeros((self.adata.shape[0], self.p_dim), dtype=np.float32)
        for cond in tqdm(np.unique(self.adata.obs['condition'].values)):
            if cond == 'ctrl':
                continue
            if cond in self.gene_emb:
                emb_matrix[self.adata.obs['condition'].values == cond] = self.gene_emb[cond]
        self.adata.obsm['pert_emb'] = emb_matrix

        gc.collect()
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Context data builder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_context_data(self, context: str):
        """
        Build matched (ctrl, pert_emb, true_delta, condition_id) tensors
        for one context.  Holdout perts are excluded.
        """
        mask = self.adata.obs['context_cell'] == context
        data = self.adata[mask]

        datalist = []
        for cond in np.unique(data.obs['condition'].values):
            if cond == 'ctrl':
                continue
            if cond not in self.gene_emb:           # no embedding → skip
                continue
            if cond in self._holdout_names:          # holdout → skip for training
                continue

            cond_mask    = data.obs['condition'] == cond
            cond_data    = data[cond_mask]           # only contexts that have this cond

            for ctx_id in np.unique(cond_data.obs['context_cell'].values):
                ctrl_mask_local = (
                    (data.obs['context_cell'] == ctx_id) &
                    (data.obs['condition'] == 'ctrl')
                )
                pert_mask_local = (
                    (data.obs['context_cell'] == ctx_id) &
                    (data.obs['condition'] == cond)
                )

                ctrl_cells = data[ctrl_mask_local].X
                pert_cells = data[pert_mask_local].layers['normalized_exp']
                pert_embs  = data[pert_mask_local].obsm['pert_emb']

                if ctrl_cells.shape[0] == 0 or pert_cells.shape[0] == 0:
                    continue

                n_cells      = pert_cells.shape[0]
                ctrl_idx     = np.random.choice(ctrl_cells.shape[0], n_cells, replace=True)
                matched_ctrl = ctrl_cells[ctrl_idx]

                datalist.append((
                    np.array(matched_ctrl),
                    np.array(pert_embs),
                    np.array(pert_cells),
                    np.tile(self.pert_encoder[cond], (n_cells, 1)),
                ))

                del ctrl_cells, pert_cells, matched_ctrl, ctrl_idx
                gc.collect()

        if not datalist:
            return None

        return (
            torch.FloatTensor(np.vstack([b[0] for b in datalist])),
            torch.FloatTensor(np.vstack([b[1] for b in datalist])),
            torch.FloatTensor(np.vstack([b[2] for b in datalist])),
            torch.LongTensor( np.vstack([b[3] for b in datalist])),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dataloader preparation
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_dataloaders(self, train_contexts: List[str]):
        """Load or compute and cache per-context dataloaders."""
        self._context_dataloaders = {}

        for ctx in train_contexts:
            cache_file = os.path.join(self.cache_dir, f"context_{ctx}.pkl")

            if os.path.exists(cache_file):
                print(f"  [cache] Loading {ctx}...")
                with open(cache_file, 'rb') as f:
                    d = pickle.load(f)
                ctrl_exp, pert_embs, pert_exp, pert_cond = (
                    d['ctrl_exp'], d['pert_embs'], d['pert_exp'], d['pert_conditions']
                )
            else:
                print(f"  [data]  Generating {ctx}...")
                result = self._get_context_data(ctx)
                if result is None:
                    print(f"  [warn]  {ctx} has no valid pert data after filtering. Skipping.")
                    continue
                ctrl_exp, pert_embs, pert_exp, pert_cond = result
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'ctrl_exp': ctrl_exp,
                        'pert_embs': pert_embs,
                        'pert_exp': pert_exp,
                        'pert_conditions': pert_cond,
                    }, f)

            # Filter holdouts from loaded data
            if self._holdout_ids:
                ctrl_exp, pert_embs, pert_exp, pert_cond = self._filter_holdouts(
                    ctrl_exp, pert_embs, pert_exp, pert_cond
                )
                if ctrl_exp.shape[0] == 0:
                    print(f"  [warn]  {ctx} empty after holdout filtering. Skipping.")
                    continue

            dataset = PerturbDataset(ctrl_exp, pert_embs, pert_exp, pert_cond)
            self._context_dataloaders[ctx] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_min2,
                num_workers=0,
            )

        gc.collect()
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Holdout filtering
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_holdouts(self, ctrl_exp, pert_embs, pert_exp, pert_conditions):
        """Remove rows whose condition ID is in self._holdout_ids."""
        if not self._holdout_ids:
            return ctrl_exp, pert_embs, pert_exp, pert_conditions

        cond_1d  = pert_conditions.view(-1).to(dtype=torch.long)
        keep     = ~torch.isin(
            cond_1d,
            torch.tensor(list(self._holdout_ids), dtype=torch.long)
        )
        return (
            ctrl_exp[keep],
            pert_embs[keep],
            pert_exp[keep],
            cond_1d[keep].unsqueeze(1),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _loss(pred, true):
        return 0.5 * torch.mean(torch.sum((pred - true) ** 2, dim=1))

    # ─────────────────────────────────────────────────────────────────────────
    # Net factory
    # ─────────────────────────────────────────────────────────────────────────

    def _new_net(self) -> Net:
        return Net(
            x_dim=self.x_dim,
            p_dim=self.p_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core training loop (shared across modes)
    # ─────────────────────────────────────────────────────────────────────────

    def _train_loop(self, train_contexts: List[str]) -> int:
        """
        MAML-style meta-training over the supplied training contexts.
        Modifies self.Net in-place.
        Returns total number of NaN epochs skipped.
        """
        n_ctx = len(train_contexts)
        use_replace = n_ctx <= self.meta_batch_size * 2
        if use_replace:
            print(
                f"[HyperMAP] WARNING: {n_ctx} training context(s) available — "
                f"sampling with replacement to fill meta_batch_size={self.meta_batch_size}. "
                "For best results, provide more reference contexts (ideally > 8)."
            )

        meta_optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.meta_lr)
        gene_coverage: Dict[int, int] = {}
        nan_epochs = 0
        consecutive_nan = 0

        self.Net.train()

        for epoch in tqdm(range(self.training_epochs), desc="Training"):
            meta_optimizer.zero_grad()
            task_losses = []

            meta_batch = np.random.choice(
                train_contexts, size=self.meta_batch_size, replace=use_replace
            )

            for ctx in meta_batch:
                if ctx not in self._context_dataloaders:
                    continue

                dataloader    = self._context_dataloaders[ctx]
                dl_iter       = iter(dataloader)
                n_batches     = 5  # inner-loop batches per context

                with higher.innerloop_ctx(
                    self.Net,
                    torch.optim.SGD(self.Net.parameters(), lr=self.inner_lr),
                    copy_initial_weights=False,
                ) as (fnet, diffopt):

                    # ── Inner loop ────────────────────────────────────────
                    for _ in range(n_batches):
                        try:
                            cb, pb, tb, cond_b = next(dl_iter)
                        except StopIteration:
                            dl_iter = iter(dataloader)
                            cb, pb, tb, cond_b = next(dl_iter)

                        cb, pb, tb = cb.to(self.device), pb.to(self.device), tb.to(self.device)

                        for g in torch.unique(cond_b).cpu().numpy():
                            gene_coverage[int(g)] = gene_coverage.get(int(g), 0) + 1

                        for _ in range(self.n_inner_steps):
                            pred, _, _ = fnet(cb, pb)
                            loss = self._loss(pred, tb)
                            diffopt.step(loss)

                        del pred, cb, pb, tb, loss
                        torch.cuda.empty_cache()

                    # ── Outer loss ────────────────────────────────────────
                    try:
                        cb, pb, tb, _ = next(dl_iter)
                    except StopIteration:
                        dl_iter = iter(dataloader)
                        cb, pb, tb, _ = next(dl_iter)

                    cb, pb, tb = cb.to(self.device), pb.to(self.device), tb.to(self.device)
                    pred, _, _ = fnet(cb, pb)
                    outer_loss = self._loss(pred, tb)
                    task_losses.append(outer_loss)

                    del pred, cb, pb, tb, outer_loss

            if not task_losses:
                print(f"  Epoch {epoch+1}: no task losses (check dataloaders).")
                continue

            meta_loss = torch.stack(task_losses).mean()

            # ── NaN/Inf check — skip update, keep weights clean ───────────
            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                nan_epochs += 1
                consecutive_nan += 1
                print(f"  Epoch {epoch+1}: meta-loss is NaN/Inf — skipping update "
                      f"({nan_epochs} total skipped).")
                del task_losses, meta_loss
                gc.collect()
                torch.cuda.empty_cache()
                continue

            consecutive_nan = 0  # reset on clean epoch
            meta_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.Net.parameters(), self.grad_clip)
            meta_optimizer.step()
            del task_losses
            torch.cuda.empty_cache()

            print(f"  Epoch {epoch+1}/{self.training_epochs}  meta-loss: {meta_loss.item():.4f}")

            if (epoch + 1) % 10 == 0:
                total = len(self.pert_encoder) - 1
                seen  = len(gene_coverage)
                print(f"  Gene coverage: {seen}/{total} ({100*seen/max(total,1):.1f}%)")

        return nan_epochs

    # ─────────────────────────────────────────────────────────────────────────
    # Adaptation gene selection
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_least_consistent(self, train_contexts: List[str]) -> torch.Tensor:
        """
        Compute per-pert coefficient of variation across training contexts.
        Matches original _compute_gene_statistics logic exactly:
            - mean_effect = mean(abs(mean(stacked, dim=0)))  per pert
            - effect_std  = sqrt(mean(var(stacked, dim=0)))  per pert
            - CV          = effect_std / abs(mean_effect)
        Returns pert integer IDs sorted by CV descending (most variable first).
        """
        device = self.device
        all_perts = [k for k in self.pert_encoder.keys() if k != 'ctrl'
                     and k not in self._holdout_names]

        mean_effects = torch.zeros(len(all_perts), device=device)
        effect_stds  = torch.zeros(len(all_perts), device=device)

        for pert_idx, pert in enumerate(all_perts):
            cell_means = []
            for ctx in train_contexts:
                mask = (
                    (self.adata.obs['context_cell'] == ctx) &
                    (self.adata.obs['condition'] == pert)
                )
                if np.sum(mask) > 0:
                    pert_effects = self.adata[mask].layers['normalized_exp']
                    cell_mean = torch.tensor(
                        np.mean(pert_effects, axis=0), device=device
                    )
                    cell_means.append(cell_mean)

            if len(cell_means) >= 2:
                stacked = torch.stack(cell_means)  # [n_contexts, n_genes]
                mean_effects[pert_idx] = torch.mean(
                    torch.abs(torch.mean(stacked, dim=0))
                )
                effect_stds[pert_idx] = torch.sqrt(
                    torch.mean(torch.var(stacked, dim=0))
                )

        # CV = std / abs(mean), only for non-zero mean perts
        non_zero = mean_effects != 0
        all_pert_ids = torch.tensor(
            [self.pert_encoder[p] for p in all_perts], dtype=torch.long, device=device
        )

        if non_zero.sum() == 0:
            return all_pert_ids  # fallback — no stats available

        cv = effect_stds[non_zero] / torch.abs(mean_effects[non_zero])
        # Sort descending — most variable (least consistent) first
        sorted_indices = torch.argsort(cv, descending=True)
        return all_pert_ids[non_zero][sorted_indices]

    def _select_adapt_perts(
        self,
        seen_ids: torch.Tensor,
        train_contexts: List[str],
    ) -> torch.Tensor:
        """
        Select which pert IDs to use for adaptation based on selection_strategy.

        random          : first n_adapt_genes from seen_ids (sorted by int ID)
        least_consistent: top n_adapt_genes by CV across training contexts
        functional      : perts supplied by user via functional_perts
        """
        n = self.n_adapt_genes

        if self.selection_strategy == 'random':
            return seen_ids[:n]

        elif self.selection_strategy == 'least_consistent':
            # Compute or reuse cached ranking
            if self._least_consistent_cache is None:
                ranked = self._compute_least_consistent(train_contexts)
                self._least_consistent_cache = ranked

            ranked = self._least_consistent_cache
            # Keep only IDs that are actually in seen_ids
            seen_set = set(seen_ids.tolist())
            selected = [int(x) for x in ranked if int(x) in seen_set][:n]
            if not selected:
                return seen_ids[:n]  # fallback
            return torch.tensor(selected, dtype=torch.long)

        elif self.selection_strategy == 'functional':
            # Map user-supplied pert names to IDs, intersect with seen
            func_ids = torch.tensor(
                [self.pert_encoder[p] for p in self.functional_perts
                 if p in self.pert_encoder],
                dtype=torch.long
            )
            seen_set = set(seen_ids.tolist())
            selected = [int(x) for x in func_ids if int(x) in seen_set][:n]
            if not selected:
                warnings.warn(
                    f"None of the functional_perts were found in this context's "
                    f"seen perturbations. Falling back to random selection.",
                    UserWarning
                )
                return seen_ids[:n]
            return torch.tensor(selected, dtype=torch.long)

        else:
            return seen_ids[:n]

    # ─────────────────────────────────────────────────────────────────────────
    # Test-time adaptation
    # ─────────────────────────────────────────────────────────────────────────

    def _adapt(self, context: str, ctrl_exp, pert_embs, true_pert, pert_conditions,
               train_contexts: Optional[List[str]] = None):
        """
        Few-shot adaptation to a new context using a subset of seen perturbations.
        Holdouts are excluded from adaptation data.

        Returns an adapted model (higher functional net).
        """
        print(f"  [adapt] {context}  strategy={self.selection_strategy}  "
              f"n_adapt_genes={self.n_adapt_genes}...")

        cond_1d = pert_conditions.squeeze(1)

        # Exclude holdouts from adaptation
        seen_ids = torch.unique(cond_1d)
        if self._holdout_ids:
            holdout_t = torch.tensor(list(self._holdout_ids), dtype=torch.long)
            seen_ids  = seen_ids[~torch.isin(seen_ids, holdout_t)]

        if seen_ids.numel() == 0:
            warnings.warn(
                f"[{context}] No non-holdout perts available for adaptation. "
                "Using base meta-learned model.",
                UserWarning
            )
            return copy.deepcopy(self.Net).to(self.device)

        # ── Select perts for adaptation ────────────────────────────────────────
        train_ctxs = train_contexts or []
        selected    = self._select_adapt_perts(seen_ids, train_ctxs)
        subset_mask = torch.isin(cond_1d, selected)

        sub_ctrl = ctrl_exp[subset_mask]
        sub_pemb = pert_embs[subset_mask]
        sub_true = true_pert[subset_mask]
        sub_cond = pert_conditions[subset_mask]

        # Adaptation data guard
        n = sub_ctrl.shape[0]
        status = check_adaptation_data(n, context)
        if status == 'skip':
            return copy.deepcopy(self.Net).to(self.device)

        batch_size = max(2, min(100, n // 2))
        adapt_ds   = PerturbDataset(sub_ctrl, sub_pemb, sub_true, sub_cond)
        adapt_dl   = DataLoader(
            adapt_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_min2,
            num_workers=0,
        )

        adapted_model  = copy.deepcopy(self.Net).to(self.device)
        inner_opt      = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr_adapt)
        adapted_model.train()

        with higher.innerloop_ctx(
            adapted_model,
            inner_opt,
            copy_initial_weights=True,
        ) as (fnet, diffopt):

            dl_iter  = iter(adapt_dl)
            n_batches = len(adapt_dl)

            for batch_idx in range(n_batches):
                try:
                    cb, pb, tb, _ = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(adapt_dl)
                    cb, pb, tb, _ = next(dl_iter)

                cb, pb, tb = cb.to(self.device), pb.to(self.device), tb.to(self.device)

                for step in range(self.n_adapt_steps):
                    pred, _, _ = fnet(cb, pb)
                    loss = self._loss(pred, tb)

                    # ── Adaptation explosion check ────────────────────────
                    if torch.isnan(loss) or torch.isinf(loss):
                        del pred, cb, pb, tb, loss
                        gc.collect()
                        torch.cuda.empty_cache()
                        return None  # caller handles skip + summary

                    diffopt.step(loss)

                    if batch_idx == 0 and step == 0:
                        print(f"    initial adapt loss: {loss.item():.4f}")
                    if batch_idx == n_batches - 1 and step == self.n_adapt_steps - 1:
                        print(f"    final adapt loss:   {loss.item():.4f}")

                del pred, cb, pb, tb, loss
                torch.cuda.empty_cache()

            adapted_model = fnet

        adapted_model.eval()
        return adapted_model

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_from_data(
        self,
        adapted_model,
        ctrl_exp,
        pert_embs,
        pert_conditions,
    ):
        """
        Run forward pass in batches. Returns pred tensor on CPU.
        Used in LOO (predict perts present in training data).
        """
        eval_ds = TensorDataset(ctrl_exp, pert_embs, pert_conditions)
        eval_dl = DataLoader(eval_ds, batch_size=4096, shuffle=False)

        all_preds = []
        adapted_model.eval()
        with torch.no_grad():
            for bc, bp, _ in eval_dl:
                bc, bp = bc.to(self.device), bp.to(self.device)
                pred, _, _ = adapted_model(bc, bp)
                all_preds.append(pred.cpu())
                del bc, bp, pred
                torch.cuda.empty_cache()

        return torch.cat(all_preds, dim=0)

    def _predict_from_gene_emb(
        self,
        adapted_model,
        context: str,
        pert_names: List[str],
        n_ctrl_samples: int = 20,
    ):
        """
        Predict ALL perts in pert_names using gene_emb embeddings.
        Controls are sampled from this context's ctrl cells.

        Returns dict { pert_name: np.ndarray [n_ctrl_samples, n_genes] }
        """
        ctrl_mask = (
            (self.adata.obs['context_cell'] == context) &
            (self.adata.obs['condition'] == 'ctrl')
        )
        ctrl_X = self.adata[ctrl_mask].X.astype(np.float32)
        replace = ctrl_X.shape[0] < n_ctrl_samples
        ctrl_sampled = ctrl_X[
            np.random.choice(ctrl_X.shape[0], n_ctrl_samples, replace=replace)
        ]

        n_perts = len(pert_names)
        ctrl_tensor = torch.from_numpy(
            np.repeat(ctrl_sampled, n_perts, axis=0)
        ).float()
        pert_emb_matrix = np.vstack(
            [self.gene_emb[p] for p in pert_names]
        ).astype(np.float32)
        pert_tensor = torch.from_numpy(
            np.tile(pert_emb_matrix, (n_ctrl_samples, 1))
        ).float()

        adapted_model.eval()
        adapted_model = adapted_model.to("cpu")

        all_preds = []
        infer_batch = 2048
        with torch.no_grad():
            for start in range(0, ctrl_tensor.shape[0], infer_batch):
                end  = start + infer_batch
                pred, _, _ = adapted_model(ctrl_tensor[start:end], pert_tensor[start:end])
                all_preds.append(pred.cpu())

        predicted = torch.cat(all_preds, dim=0)  # [n_ctrl*n_perts, genes]
        predicted = predicted.view(n_ctrl_samples, n_perts, -1).permute(1, 0, 2)
        # → [n_perts, n_ctrl_samples, genes]

        return {
            p: predicted[i].numpy()
            for i, p in enumerate(pert_names)
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Model save / load
    # ─────────────────────────────────────────────────────────────────────────

    def _save_model(self, tag: str):
        """Save current self.Net state dict to cache dir."""
        path = os.path.join(self.cache_dir, f"model_{tag}.pt")
        torch.save(self.Net.state_dict(), path)
        print(f"  [save] Model saved → {path}")

    def load_model(self, path: str):
        """Load a saved model state dict."""
        self.Net = self._new_net()
        self.Net.load_state_dict(torch.load(path, map_location=self.device))
        self.Net.to(self.device)
        print(f"[HyperMAP] Model loaded from {path}")

    def _print_summary(
        self,
        nan_epochs:      int,
        skipped_contexts: List[str],
        total_contexts:  int,
    ):
        """Print end-of-run summary for meta-training NaN and adaptation failures."""
        print("\n" + "="*60)
        print("[HyperMAP] Run complete.")

        # ── Meta-training summary ─────────────────────────────────────────
        clean_epochs = self.training_epochs - nan_epochs
        if nan_epochs == 0:
            print(f"  Meta-training : {clean_epochs}/{self.training_epochs} epochs clean.")
        else:
            print(f"  Meta-training : {clean_epochs}/{self.training_epochs} epochs clean. "
                  f"{nan_epochs} NaN/Inf epochs skipped.")
            if nan_epochs > 5:
                print(f"  ⚠  More than 5 NaN epochs detected. Training may be unstable.")
                print(f"     Recommendation: rerun with lower inner_lr — "
                      f"try {round(self.inner_lr * 0.8, 5)} (current {self.inner_lr})")

        # ── Adaptation summary ────────────────────────────────────────────
        n_ok = total_contexts - len(skipped_contexts)
        if not skipped_contexts:
            print(f"  Adaptation    : all {total_contexts} context(s) completed successfully.")
        else:
            print(f"  Adaptation    : {n_ok}/{total_contexts} context(s) completed.")
            print(f"  Failed        : {skipped_contexts}")
            print(f"  ⚠  Recommendation: rerun failed context(s) with either:")
            print(f"     - lower inner_lr_adapt — "
                  f"try {round(self.inner_lr_adapt * 0.8, 5)} (current {self.inner_lr_adapt})")
            print(f"     - fewer n_adapt_steps  — "
                  f"try {max(1, self.n_adapt_steps - 1)} (current {self.n_adapt_steps})")
        print("="*60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # LOO evaluation helper
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_loo_context(
        self,
        context: str,
        train_perts: set,
        train_contexts: List[str],
        prediction_mode: Literal['original', 'sampled'] = 'original',
    ) -> Dict:
        """
        Adapt and evaluate one left-out context.

        prediction_mode : 'original' or 'sampled'
            'original' : feeds cached ctrl cells through the model, predicts only
                         perts measured in the test context, averages over all cells.
                         Faithful reproduction of adapt_and_evaluate(). Default.
            'sampled'  : samples 20 fresh ctrl cells from adata, predicts all
                         train_perts + holdouts regardless of whether measured
                         in the test context. true_delta is None for unmeasured perts.
        """
        # ── Load context data ─────────────────────────────────────────────────
        # If holdouts are defined we must use _get_context_data_full so that
        # holdout perts are present for ground-truth lookup.
        # Otherwise load from cache (faster).
        if self._holdout_names:
            result = self._get_context_data_full(context)
            if result is None:
                return {}
            ctrl_exp, pert_embs, true_pert, pert_conditions = result
        else:
            cache_file = os.path.join(self.cache_dir, f"context_{context}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    d = pickle.load(f)
                ctrl_exp        = d['ctrl_exp']
                pert_embs       = d['pert_embs']
                true_pert       = d['pert_exp']
                pert_conditions = d['pert_conditions']
            else:
                result = self._get_context_data_full(context)
                if result is None:
                    return {}
                ctrl_exp, pert_embs, true_pert, pert_conditions = result

        # ── Adapt (holdouts excluded from adaptation data) ────────────────────
        adapted = self._adapt(context, ctrl_exp, pert_embs, true_pert, pert_conditions,
                              train_contexts=train_contexts)
        if adapted is None:
            return None

        cond_1d = pert_conditions.squeeze(1)
        results = {}

        # ── 'original' mode ───────────────────────────────────────────────────
        # Feeds cached ctrl_exp through model. Predicts only perts actually
        # present in this context. Averages over all cells in the cache.
        if prediction_mode == 'original':
            pred_pert = self._predict_from_data(adapted, ctrl_exp, pert_embs, pert_conditions)

            unique_ids = torch.unique(cond_1d).tolist()
            for pert_id in unique_ids:
                pert_id = int(pert_id)
                cond = next((k for k, v in self.pert_encoder.items() if v == pert_id), None)
                if cond is None or cond == 'ctrl':
                    continue
                if cond not in self.gene_emb:
                    continue

                pert_mask = (cond_1d == pert_id).squeeze()
                if not torch.any(pert_mask):
                    continue

                results[cond] = {
                    'pred_delta': pred_pert[pert_mask].mean(dim=0).numpy(),
                    'true_delta': true_pert[pert_mask].mean(dim=0).numpy(),
                }

        # ── 'sampled' mode ────────────────────────────────────────────────────
        # Samples 20 fresh ctrl cells. Predicts all train_perts + holdouts.
        # true_delta is None for perts not measured in this context.
        elif prediction_mode == 'sampled':
            valid_perts = [
                p for p in train_perts
                if p in self.gene_emb and p != 'ctrl'
            ]
            holdout_pred_perts = [
                p for p in self._holdout_names
                if p in self.gene_emb and p != 'ctrl'
            ]
            all_pred_perts = valid_perts + holdout_pred_perts
            pred_dict = self._predict_from_gene_emb(
                adapted, context, all_pred_perts, n_ctrl_samples=20
            )

            for pert in all_pred_perts:
                pred_delta = pred_dict[pert].mean(axis=0)

                true_delta = None
                if pert in self.pert_encoder:
                    pert_id   = self.pert_encoder[pert]
                    pert_mask = (cond_1d == pert_id).squeeze()
                    if torch.any(pert_mask):
                        true_delta = true_pert[pert_mask].mean(dim=0).numpy()

                results[pert] = {
                    'pred_delta': pred_delta,
                    'true_delta': true_delta,
                }

        del adapted
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _get_context_data_full(self, context: str):
        """
        Like _get_context_data but includes ALL perts (incl. holdouts).
        Used for ground-truth lookup during evaluation only — NOT for training.
        """
        mask = self.adata.obs['context_cell'] == context
        data = self.adata[mask]
        datalist = []

        for cond in np.unique(data.obs['condition'].values):
            if cond == 'ctrl':
                continue
            if cond not in self.gene_emb:
                continue
            # NOTE: holdouts ARE included here for eval ground truth

            if cond not in self.pert_encoder:
                warnings.warn(
                    f"[{context}] Condition '{cond}' not in pert_encoder — skipping. "
                    "This should not happen if adata was not modified after init.",
                    UserWarning
                )
                continue

            cond_mask = data.obs['condition'] == cond
            cond_data = data[cond_mask]             # only contexts that have this cond

            for ctx_id in np.unique(cond_data.obs['context_cell'].values):
                ctrl_m = (data.obs['context_cell'] == ctx_id) & (data.obs['condition'] == 'ctrl')
                pert_m = (data.obs['context_cell'] == ctx_id) & (data.obs['condition'] == cond)

                ctrl_cells = data[ctrl_m].X
                pert_cells = data[pert_m].layers['normalized_exp']
                pert_embs  = data[pert_m].obsm['pert_emb']

                if ctrl_cells.shape[0] == 0 or pert_cells.shape[0] == 0:
                    continue

                n_cells  = pert_cells.shape[0]
                ctrl_idx = np.random.choice(ctrl_cells.shape[0], n_cells, replace=True)

                datalist.append((
                    np.array(ctrl_cells[ctrl_idx]),
                    np.array(pert_embs),
                    np.array(pert_cells),
                    np.tile(self.pert_encoder[cond], (n_cells, 1)),
                ))
                del ctrl_cells, pert_cells, ctrl_idx
                gc.collect()

        if not datalist:
            return None
        return (
            torch.FloatTensor(np.vstack([b[0] for b in datalist])),
            torch.FloatTensor(np.vstack([b[1] for b in datalist])),
            torch.FloatTensor(np.vstack([b[2] for b in datalist])),
            torch.LongTensor( np.vstack([b[3] for b in datalist])),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ── PUBLIC API ────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    def loo(
        self,
        contexts: Optional[List[str]] = None,
        prediction_mode: Literal['original', 'sampled'] = 'original',
    ) -> Dict:
        """
        Leave-one-context-out cross-validation.

        For every context:
            - Trains on all other contexts
            - Adapts to the left-out context using n_adapt_genes seen perturbations
            - Predicts perturbations according to prediction_mode
            - true_delta is None for perts not measured in the left-out context

        Parameters
        ----------
        contexts : list of str, optional
            Specific contexts to use as test folds.
            If None, all contexts in adata are used (standard full LOO).
            Training always uses all contexts except the current test context.

        prediction_mode : 'original' or 'sampled'
            'original' : feeds cached ctrl cells through the model, predicts only
                         perts measured in the test context, averages over all cells.
                         Faithful reproduction of adapt_and_evaluate(). Default.
            'sampled'  : samples 20 fresh ctrl cells from adata, predicts all
                         train_perts + holdouts regardless of whether measured
                         in the test context. true_delta is None for unmeasured perts.

        Returns
        -------
        dict
            {
              context_name: {
                pert_name: {
                  'pred_delta': np.ndarray  (n_genes,),
                  'true_delta': np.ndarray or None
                }
              }
            }
        """
        all_contexts = list(np.unique(self.adata.obs['context_cell']))

        if contexts is not None:
            # Validate requested contexts exist
            missing = set(contexts) - set(all_contexts)
            if missing:
                raise ValueError(
                    f"The following contexts were not found in adata:\n  {sorted(missing)}"
                )
            test_contexts = contexts
        else:
            test_contexts = all_contexts

        all_results = {}
        skipped_contexts: List[str] = []
        nan_epochs = 0

        for test_ctx in test_contexts:
            print(f"\n{'='*60}")
            print(f"[LOO] Test context: {test_ctx}")

            # Training always uses ALL contexts except the current test one
            train_contexts = [c for c in all_contexts if c != test_ctx]

            # Perts present in training data (union, excluding ctrl + holdouts)
            train_perts = set()
            for ctx in train_contexts:
                ctx_mask = (
                    (self.adata.obs['context_cell'] == ctx) &
                    (self.adata.obs['condition'] != 'ctrl')
                )
                for p in self.adata.obs.loc[ctx_mask, 'condition'].unique():
                    if p not in self._holdout_names and p in self.gene_emb:
                        train_perts.add(p)

            self.Net = self._new_net()
            self._prepare_dataloaders(train_contexts)
            nan_epochs += self._train_loop(train_contexts)
            self._save_model(f"loo_{test_ctx}")

            # Reset per-fold cache for least_consistent strategy
            self._least_consistent_cache = None

            print(f"\n[LOO] Evaluating {test_ctx}...")
            ctx_results = self._evaluate_loo_context(
                test_ctx, train_perts, train_contexts,
                prediction_mode=prediction_mode,
            )

            if ctx_results is None:
                print(f"  [skip] {test_ctx} — adaptation failed (NaN/Inf).")
                all_results[test_ctx] = None
                skipped_contexts.append(test_ctx)
            else:
                all_results[test_ctx] = ctx_results
                seen  = sum(1 for v in ctx_results.values() if v['true_delta'] is not None)
                total = len(ctx_results)
                print(f"  Perts predicted: {total}  |  with ground truth: {seen}  |  NA: {total-seen}")

            gc.collect()
            torch.cuda.empty_cache()

        self._print_summary(nan_epochs, skipped_contexts, len(test_contexts))
        return all_results

    # ─────────────────────────────────────────────────────────────────────────

    def train_predict(
        self,
        predict_contexts:  List[str],
        use_gene_emb_perts: bool          = False,
        output_format:      Literal['pseudobulk', 'singlecell'] = 'pseudobulk',
        n_cells_per_pert:   int            = 20,
    ):
        """
        Train on all non-predict contexts, adapt to each predict context,
        then produce expression predictions.

        Parameters
        ----------
        predict_contexts : list of str
            Context names to predict. All others are used for training.

        use_gene_emb_perts : bool
            False  → predict only perts present in training data (with NA ground truth
                     for perts missing from a given predict context).
            True   → predict ALL perts in gene_emb (includes training perts + any extras).

        output_format : 'pseudobulk' or 'singlecell'
            Pseudobulk returns one row per (context, pert) with mean predicted delta.
            Singlecell returns n_cells_per_pert rows per (context, pert).

        n_cells_per_pert : int
            Only used when output_format='singlecell'. Default 20.

        Returns
        -------
        AnnData
            obs columns: context_cell, perturbation
            X: predicted delta expression
        """
        validate_predict_contexts(self.adata, predict_contexts)

        all_contexts   = list(np.unique(self.adata.obs['context_cell']))
        train_contexts = [c for c in all_contexts if c not in predict_contexts]

        print(f"\n[train_predict] Training contexts: {train_contexts}")
        print(f"[train_predict] Predict contexts:  {predict_contexts}")

        # Training perts (for when use_gene_emb_perts=False)
        train_perts = set()
        for ctx in train_contexts:
            ctx_mask = (
                (self.adata.obs['context_cell'] == ctx) &
                (self.adata.obs['condition'] != 'ctrl')
            )
            for p in self.adata.obs.loc[ctx_mask, 'condition'].unique():
                if p not in self._holdout_names and p in self.gene_emb:
                    train_perts.add(p)

        # Perts to predict
        if use_gene_emb_perts:
            pred_perts = [p for p in self.gene_emb if p != 'ctrl']
        else:
            pred_perts = sorted(train_perts)

        print(f"[train_predict] Perturbations to predict: {len(pred_perts)}")

        # ── Train ─────────────────────────────────────────────────────────────
        self.Net = self._new_net()
        self._prepare_dataloaders(train_contexts)
        nan_epochs = self._train_loop(train_contexts)
        self._save_model("train_predict")
        self._least_consistent_cache = None

        # ── Adapt and predict for each predict context ─────────────────────────
        all_predictions: Dict = {}
        skipped_contexts: List[str] = []

        for ctx in predict_contexts:
            print(f"\n[train_predict] Predicting for: {ctx}")

            cache_file = os.path.join(self.cache_dir, f"context_{ctx}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    d = pickle.load(f)
                ctrl_exp = d['ctrl_exp']
                pert_embs = d['pert_embs']
                true_pert = d['pert_exp']
                pert_cond = d['pert_conditions']
            else:
                result = self._get_context_data_full(ctx)
                if result is None:
                    print(f"  [warn] No data for {ctx}. Skipping.")
                    skipped_contexts.append(ctx)
                    continue
                ctrl_exp, pert_embs, true_pert, pert_cond = result

            adapted = self._adapt(ctx, ctrl_exp, pert_embs, true_pert, pert_cond,
                                  train_contexts=train_contexts)
            if adapted is None:
                print(f"  [skip] {ctx} — adaptation failed (NaN/Inf).")
                skipped_contexts.append(ctx)
                continue
            pred_dict = self._predict_from_gene_emb(
                adapted, ctx, pred_perts, n_ctrl_samples=n_cells_per_pert
            )

            # Store as { pert: { pred_delta, true_delta } }
            cond_1d = pert_cond.squeeze(1)
            ctx_result = {}
            for pert in pred_perts:
                pred_mat = pred_dict[pert]  # [n_cells_per_pert, n_genes]

                # Ground truth
                true_delta = None
                if pert in self.pert_encoder:
                    pid   = self.pert_encoder[pert]
                    pmask = (cond_1d == pid).squeeze()
                    if torch.any(pmask):
                        true_delta = true_pert[pmask].mean(dim=0).numpy()

                ctx_result[pert] = {
                    'pred_delta': pred_mat,
                    'true_delta': true_delta,
                }

            all_predictions[ctx] = ctx_result
            del adapted
            gc.collect()
            torch.cuda.empty_cache()

        self._print_summary(nan_epochs, skipped_contexts, len(predict_contexts))

        # ── Format output ──────────────────────────────────────────────────────
        if output_format == 'pseudobulk':
            return build_pseudobulk_adata(all_predictions, self.adata.var_names)
        else:
            return build_singlecell_adata(
                all_predictions, self.adata.var_names, n_cells=n_cells_per_pert
            )

    # ─────────────────────────────────────────────────────────────────────────

    def impute(
        self,
        impute_contexts:    Optional[List[str]] = None,
        use_gene_emb_perts: bool                = False,
        output_format:      Literal['pseudobulk', 'singlecell'] = 'pseudobulk',
        n_cells_per_pert:   int                 = 20,
    ):
        """
        Train on ALL contexts, then adapt each context and predict perturbations.

        Parameters
        ----------
        impute_contexts : list of str, optional
            Contexts to produce predictions for.
            If None, all contexts are used.

        use_gene_emb_perts : bool
            False → predict only perts present in the full dataset.
            True  → predict all perts in gene_emb.

        output_format : 'pseudobulk' or 'singlecell'

        n_cells_per_pert : int
            Only used when output_format='singlecell'. Default 20.

        Returns
        -------
        AnnData
        """
        all_contexts = list(np.unique(self.adata.obs['context_cell']))
        if impute_contexts is None:
            impute_contexts = all_contexts

        print(f"\n[impute] Training on ALL {len(all_contexts)} contexts")
        print(f"[impute] Imputing for: {impute_contexts}")

        # Perts to predict
        if use_gene_emb_perts:
            pred_perts = [p for p in self.gene_emb if p != 'ctrl']
        else:
            all_perts = set()
            non_ctrl = self.adata.obs['condition'] != 'ctrl'
            for p in self.adata.obs.loc[non_ctrl, 'condition'].unique():
                if p not in self._holdout_names and p in self.gene_emb:
                    all_perts.add(p)
            pred_perts = sorted(all_perts)

        print(f"[impute] Perturbations to predict: {len(pred_perts)}")

        # ── Train on all contexts ──────────────────────────────────────────────
        self.Net = self._new_net()
        self._prepare_dataloaders(all_contexts)
        nan_epochs = self._train_loop(all_contexts)
        self._save_model("impute")
        self._least_consistent_cache = None

        # ── Adapt and predict per context ──────────────────────────────────────
        all_predictions: Dict = {}
        skipped_contexts: List[str] = []

        for ctx in impute_contexts:
            print(f"\n[impute] Predicting for: {ctx}")

            cache_file = os.path.join(self.cache_dir, f"context_{ctx}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    d = pickle.load(f)
                ctrl_exp  = d['ctrl_exp']
                pert_embs = d['pert_embs']
                true_pert = d['pert_exp']
                pert_cond = d['pert_conditions']
            else:
                result = self._get_context_data_full(ctx)
                if result is None:
                    print(f"  [warn] No data for {ctx}. Skipping.")
                    skipped_contexts.append(ctx)
                    continue
                ctrl_exp, pert_embs, true_pert, pert_cond = result

            # Determine available perts for this context (excluding holdouts)
            cond_1d  = pert_cond.squeeze(1)
            avail    = torch.unique(cond_1d).tolist()
            seen_ids = {
                int(x) for x in avail
                if int(x) not in self._holdout_ids
            }
            n_seen = len(seen_ids)
            n_adapt = max(1, min(self.n_adapt_genes, n_seen))

            if n_seen < 2:
                warnings.warn(
                    f"[{ctx}] Only {n_seen} non-holdout pert(s) available for adaptation. "
                    "Using base model.",
                    UserWarning
                )
                adapted = copy.deepcopy(self.Net).to(self.device)
            else:
                orig = self.n_adapt_genes
                self.n_adapt_genes = n_adapt
                adapted = self._adapt(ctx, ctrl_exp, pert_embs, true_pert, pert_cond,
                                      train_contexts=all_contexts)
                self.n_adapt_genes = orig

            if adapted is None:
                print(f"  [skip] {ctx} — adaptation failed (NaN/Inf).")
                skipped_contexts.append(ctx)
                continue

            pred_dict = self._predict_from_gene_emb(
                adapted, ctx, pred_perts, n_ctrl_samples=n_cells_per_pert
            )

            ctx_result = {}
            for pert in pred_perts:
                ctx_result[pert] = {'pred_delta': pred_dict[pert], 'true_delta': None}

            all_predictions[ctx] = ctx_result
            del adapted
            gc.collect()
            torch.cuda.empty_cache()

        self._print_summary(nan_epochs, skipped_contexts, len(impute_contexts))

        # ── Format output ──────────────────────────────────────────────────────
        if output_format == 'pseudobulk':
            return build_pseudobulk_adata(all_predictions, self.adata.var_names)
        else:
            return build_singlecell_adata(
                all_predictions, self.adata.var_names, n_cells=n_cells_per_pert
            )