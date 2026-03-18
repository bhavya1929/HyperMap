# HyperMAP

**Meta-learning model for single-cell perturbation prediction across biological contexts.**

HyperMAP uses a hypernetwork-based meta-learning framework to predict transcriptional responses to genetic or chemical perturbations. It is designed to generalise to unseen cell types or biological contexts through fast test-time adaptation with a small number of observed perturbations.

---

## Installation

```bash
git clone https://github.com/your-org/hypermap.git
cd hypermap
pip install -e .
```

> **PyTorch:** Install the version matching your CUDA setup from [pytorch.org](https://pytorch.org) before running the above.

**Dependencies:**
```
torch>=2.0.0
higher>=0.2.1
scanpy>=1.9.0
anndata>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

---

## Data Format

HyperMAP expects an `AnnData` object with the following structure:

| Field | Description |
|---|---|
| `adata.X` | Dense numpy array of gene expression values. Call `.toarray()` if sparse. |
| `adata.obs['context_cell']` | Biological context identifier (e.g. cell line name, donor ID). |
| `adata.obs['condition']` | Perturbation name, or `'ctrl'` for unperturbed cells. |

Every unique `context_cell` must have at least one row where `condition == 'ctrl'`.

```python
import scanpy as sc

adata = sc.read_h5ad('my_data.h5ad')
adata.X = adata.X.toarray()          # ensure dense

# Required columns
print(adata.obs['context_cell'].unique())   # e.g. ['rpe1', 'jurkat', 'k562']
print(adata.obs['condition'].unique())      # e.g. ['ctrl', 'TP53', 'MYC', ...]
```

---

## Quick Start

```python
import pickle
from hypermap import HyperMAP

# Load pre-computed perturbation embeddings (e.g. from a language model)
# Format: { pert_name: np.ndarray }
with open('gene_embeddings.pkl', 'rb') as f:
    gene_emb = pickle.load(f)

model = HyperMAP(
    adata        = adata,
    gene_emb     = gene_emb,       # pass None to use one-hot embeddings
    project_name = "my_project",
)
```

If `gene_emb` is `None`, HyperMAP automatically builds one-hot vectors from the conditions in `adata`.

---

## Three Modes

### 1. Leave-One-Context-Out Evaluation

Trains on all contexts except one, adapts to the held-out context, and evaluates predictions. Repeats for every context.

```python
loo_results = model.loo()
```

Returns a nested dict: `{ context -> { pert -> { 'pred_delta': ..., 'true_delta': ... } } }`

### 2. Train / Predict Split

Train on specified contexts, predict on others.

```python
adata_pred = model.train_predict(
    predict_contexts   = ['jurkat', 'k562'],
    use_gene_emb_perts = True,            # predict all perts in gene_emb
    output_format      = 'pseudobulk',    # or 'singlecell'
)
```

### 3. Impute

Trains on all contexts jointly and fills in missing perturbations across contexts — enabling shared transfer where one context's observations inform predictions in another.

```python
adata_imp = model.impute(
    output_format    = 'singlecell',
    n_cells_per_pert = 20,
)
```

**Output formats:**
- `'pseudobulk'` — one row per `(context, perturbation)`, values are mean predicted delta.
- `'singlecell'` — N rows per `(context, perturbation)`, values are individual predicted deltas.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `latent_dim` | 64 | Latent dimension for encoders |
| `hidden_dim` | 256 | Hidden dimension throughout the network |
| `training_epochs` | 50 | Number of meta-learning epochs |
| `batch_size` | 512 | Dataloader batch size |
| `meta_lr` | 0.0005 | Outer (meta) optimizer learning rate |
| `inner_lr` | 0.005 | Inner loop SGD learning rate |
| `n_inner_steps` | 5 | Inner gradient steps per batch during meta-training |
| `n_adapt_genes` | 10 | Number of perturbations used for test-time adaptation |
| `n_adapt_steps` | 1 | Gradient steps per batch during adaptation |
| `inner_lr_adapt` | `inner_lr` | Adaptation-specific learning rate |
| `selection_strategy` | `'random'` | How to select adaptation perturbations (see below) |
| `holdout_perts` | None | Perturbations excluded from training and adaptation |
| `seed` | 1234 | Random seed |

---

## Adaptation Gene Selection

HyperMAP adapts to a new context using a small number of observed perturbations. Three strategies are available for selecting which perturbations to use:

```python
# Default — no prior knowledge needed
model = HyperMAP(adata=adata, gene_emb=gene_emb, selection_strategy='random')

# Best when test context is biologically diverse from training
model = HyperMAP(adata=adata, gene_emb=gene_emb, selection_strategy='least_consistent')

# When you know which perturbations are informative for the test context
model = HyperMAP(
    adata              = adata,
    gene_emb           = gene_emb,
    selection_strategy = 'functional',
    functional_perts   = ['TP53', 'MYC', 'BRCA1'],
)
```

---

## License

MIT License. See `LICENSE` for details.
