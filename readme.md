# GAMMI: Graph-Guided Adversarial Mosaic Multi-omics Integration

GAMMI (Graph-Guided Adversarial Mosaic Multi-omics Integration) is a graph-based framework for integrating **single-cell and spatial multi-omics data** under mosaic settings (partially overlapping cells and features across modalities).

GAMMI represents **cells and features as nodes in a joint graph**, connects them with cell–feature and feature–feature edges, and learns low-dimensional embeddings via **contrastive learning + adversarial batch correction**.  
It supports:

- Paired & unpaired multi-omics (e.g. RNA + ATAC, RNA + ADT, RNA + methylation, etc.)
- Mosaic multi-omics across multiple datasets and technologies
- Integration of single-cell and spatial omics
- Batch effect removal and clustering-friendly embeddings
- Unified benchmarking with scIB-style metrics


---

## 1. Method Overview

### 1.1 Graph construction

Given a multi-omics dataset stored as a `MuData` object, GAMMI builds a heterogeneous graph:

- **Nodes**
  - Cell nodes: each cell from any modality (`rna`, `atac`, `adt`, `met`, `protein`, …)
  - Feature nodes: genes, peaks, proteins, etc. from each modality
- **Edges**
  - Cell–feature edges: derived from expression / accessibility matrices  
    (log-transform + min-max normalization; edge weights proportional to signal)
  - Feature–feature edges: provided as a pre-computed interaction network  
    (e.g. gene regulatory network, peak–gene links), stored in `mdata.varp["net"]`
- **Batch information**
  - A batch label is assigned to each cell node (e.g. donor, technology, library).  
    Random batch embeddings and a domain classifier are used for adversarial training.

### 1.2 Embedding model

GAMMI uses a learnable embedding layer for all nodes and optimizes it by:

- **Contrastive / ranking loss** on positive vs. negative edges  
  - Positive edges: observed cell–feature or feature–feature edges  
  - Negative edges: sampled non-edges (configurable `num_neg_per_pos`, `neg_sampling_mode`)
- **Adversarial loss** for batch effect removal  
  - A gradient-reversal layer (GRL) and domain classifier predict batch labels  
  - The encoder is trained to **fool** the batch classifier
- **Optional clustering loss**  
  - Cluster embeddings and clustering loss to encourage well-separated clusters

Key hyper-parameters (partial list):

- `num_neg_per_pos`: number of negatives for each positive edge
- `label_smoothing`: smoothing for contrastive classification loss
- `w_grad_rev`: weight of GRL (adversarial strength)
- `w_loss_cls`: weight of main contrastive loss
- `adversarial_batching_method`: how adversarial batches are sampled/organized
- `val_split`, `patience`, `num_epochs`, `learning_rate`, etc.
- `add_batch_embedding`: whether to add explicit batch embeddings
- `bilinear`: whether to use bilinear terms for edge scoring
- `adversartial_balance_weights`, `num_epochs_with_balanced_weights`: re-weighting epochs

For details, see the implementation of `GraphMosaicIntegration` in the `gmi` package.

### 1.3 Outputs

After training, GAMMI outputs:

- **Cell embeddings** (and optionally feature embeddings), saved as
  - `final_embeddings.csv`  — rows = cells, columns = embedding dimensions
- **Trained model checkpoints**
  - `model.pth`
- **UMAP visualizations** (if plotting is enabled)
  - `umap_plot_matched.png` or `umap_plot_<mode>.png`
- **Benchmarking results** (if `run_benchmark` is called)
  - `benchmark_results.csv`
  - `combined_benchmark_results_stats.csv` (across runs)
  - `scib_results.svg` (scIB-style scores)

---

## 2. Data Format Requirements

GAMMI operates on [`mudata`](https://github.com/scverse/mudata) `MuData` objects.

### 2.1 MuData structure

Each dataset is stored as:

- `mdata`: `mu.MuData` with:
  - `mdata.mod[...]`: different modalities, each an `AnnData` object  
    e.g. `rna`, `atac`, `adt`, `met`, `protein`, `spatial`
  - `mdata.varp["net"]`: feature–feature interaction matrix (sparse)  
    used as `feature_interaction_key="net"` in `GraphMosaicIntegration`
  - `mdata.obs`: global cell table (can be initially empty; we will populate it)

Typical examples:

```text
# pbmc.h5mu
MuData object with n_obs × n_vars = 17055 × 9822
  varp: 'net'
  3 modalities
    atac:    8366 × 8151   (obs: 'cluster', 'coarse_cluster', 'batch', ...)
    rna:     8689 × 1462   (obs: 'cluster', 'coarse_cluster', 'batch', ...)
    protein: 17055 × 209   (obs: 'cluster', 'coarse_cluster', 'batch', ...)

```

## 3.Installation
    
GAMMI uses uv as the environment and dependency manager.
Please ensure you have uv ≥ 0.2.0 installed:


### 3.1 Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```
### 3.2 Clone this repository

```bash
git clone https://github.com/luyiyun/graph_mosaic_integration.git
cd graph_mosaic_integration
```

### 3.3 Create & sync environment from uv.lock

```bash
uv sync
uv pip install .
```