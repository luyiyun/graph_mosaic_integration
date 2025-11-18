#!/usr/bin/env python
"""
example to reproduce a GAMMI run on PBMC.

This script:
    1) loads pbmc.h5mu
    2) runs GAMMI with baseline settings
    3) saves embeddings, UMAP, and scib-metrics results

This is the simplest reproducible example for reviewers.
"""

import os
import numpy as np
import pandas as pd
import mudata as mu
import anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

from gmi import (
    GraphMosaicIntegration,
    data_infor_integrate,
    plot_umap,
)

PBMC_PATH = "/root/autodl-tmp/pbmc.h5mu"


# ---------------------------------------------------------
# Prepare MuData
# ---------------------------------------------------------
def prepare_pbmc_mdata(mdata: mu.MuData) -> mu.MuData:
    # pull required attributes to mu-level
    mdata = data_infor_integrate(mdata, "batch", "batch", "obs")
    mdata = data_infor_integrate(mdata, "coarse_cluster", "label", "obs")
    mdata = data_infor_integrate(mdata, "lsi_pca", "Unintegrated", "obsm", dim_limit=20)

    mdata.obs["batch"] = mdata.obs["batch"].astype(str)
    return mdata


def mudata_to_anndata(mdata: mu.MuData) -> ad.AnnData:
    X = np.asarray(mdata.obsm["Unintegrated"])
    adata = ad.AnnData(X=X)
    adata.obs_names = mdata.obs_names
    adata.obs = mdata.obs[["batch", "label"]].copy()
    adata.obsm["GMI"] = mdata.obsm["GMI"]
    adata.obsm["Unintegrated"] = mdata.obsm["Unintegrated"]
    return adata


# ---------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------
def run_benchmark(mdata: mu.MuData, result_dir: str):
    adata = mudata_to_anndata(mdata)
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="label",
        embedding_obsm_keys=["GMI", "Unintegrated"],
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_leiden=True),
        batch_correction_metrics=BatchCorrection(),
    )
    bm.benchmark()
    df = bm.get_results().transpose()
    df.to_csv(os.path.join(result_dir, "benchmark_results.csv"))
    print("[OK] benchmark saved.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    result_dir = "./gammi_example_output"
    os.makedirs(result_dir, exist_ok=True)

    # 1) Load data
    mdata = mu.read(PBMC_PATH)
    mdata = prepare_pbmc_mdata(mdata)
    n_cells = mdata.obs.shape[0]

    # 2) Build GAMMI (baseline hyperparameters)
    model = GraphMosaicIntegration(
        add_batch_embedding=True,
        learning_rate=0.01,
        num_neg_per_pos=10,
        num_epochs=100,
        batch_size=131072,
        device="cuda:0",
        w_grad_rev=0.2,
        w_loss_cls=0.2,
        bilinear=True,
    )

    # 3) Fit
    feat_key = "net" if "net" in mdata.varp else None
    model.fit(mdata, batch_key="batch", feature_interaction_key=feat_key)

    # 4) Save embedding
    emb = model.embeddings.detach().cpu().numpy()[:n_cells]
    mdata.obsm["GMI"] = emb
    pd.DataFrame(emb, index=mdata.obs_names).to_csv(
        os.path.join(result_dir, "final_embeddings.csv")
    )

    # 5) UMAP
    plot_umap(mdata, result_dir)

    # 6) Benchmark
    run_benchmark(mdata, result_dir)

    print(f"Done. Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
