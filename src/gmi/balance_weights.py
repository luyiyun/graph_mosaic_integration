import logging
from typing import Literal

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.preprocessing import normalize
# import scipy.sparse as ssp


logger = logging.getLogger(__name__)


def estimate_balance_weights_glue(
    embeddings: np.ndarray,
    batch: np.ndarray,
    resolution: float = 1.0,
    cutoff: float = 0.5,
    power: float = 4.0,
    cum_operator: Literal["sum", "prod"] = "prod",
):
    assert cum_operator in ["sum", "prod"], "Invalid cum_operator!"

    logger.info("Clustering cells...")
    us, ns, leiden_labels, masks = [], [], [], []
    for bi in np.unique(batch):
        mask_bi = batch == bi
        masks.append(mask_bi)

        embed_bi = embeddings[mask_bi]
        adata_bi = ad.AnnData(
            obs=pd.DataFrame(index=np.arange(embed_bi.shape[0])),
            obsm={"embed": embed_bi},
        )
        sc.pp.neighbors(
            adata_bi,
            n_pcs=embeddings.shape[1],
            use_rep="embed",
            metric="cosine",
        )
        sc.tl.leiden(adata_bi, resolution=resolution)
        leiden_label = adata_bi.obs["leiden"].cat.codes
        leiden_labels.append(leiden_label)

        ui, ni = [], []
        for cj in np.unique(leiden_label):
            mask = leiden_label == cj
            embed_mean_ci = embed_bi[mask].mean(axis=0)
            ni.append(mask.sum())
            ui.append(embed_mean_ci)
        ui = np.stack(ui, axis=0)
        ni = np.array(ni)
        ui = normalize(ui, norm="l2")  # 保证后续计算的是cosine
        us.append(ui)
        ns.append(ni)

    logger.info("Matching clusters...")
    cosines = []
    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1 :], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            # cosine = ssp.coo_array(cosine)
            # cosine = cosine.power(power)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    if cum_operator == "sum":
        joint_cosine = 0.0
        for cosine in cosines:
            joint_cosine = joint_cosine + cosine
    elif cum_operator == "prod":
        joint_cosine = 1.0
        for cosine in cosines:
            joint_cosine = joint_cosine * cosine
    logger.info(f"Matching array shape = {joint_cosine.shape}...")

    logger.info("Estimating balancing weight...")
    weights = np.empty(embeddings.shape[0])
    for i, (ni, leiden_label, maski) in enumerate(
        zip(ns, leiden_labels, masks)
    ):
        balancing = (
            joint_cosine.sum(
                axis=tuple(k for k in range(joint_cosine.ndim) if k != i)
            )
            / ni
        )
        balancing = balancing[leiden_label]
        balancing /= balancing.mean()  # 均值点是1
        weights[maski] = balancing

    return weights
