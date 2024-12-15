import os
import numpy as np

import scanpy as sc
import mudata as mu
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation
import matplotlib.pyplot as plt
from harmony import harmonize
#优化一下复制输入部分
import anndata as ad

import numpy as np
import scipy.sparse as sp
from typing import Optional, List, Union

def get_X_from_mudata(
    mdata: mu.MuData, sparse: bool = True, fillna: float = 0.0
) -> Union[np.ndarray, sp.csr_matrix]:
    """
    Extract the combined feature matrix (X) from a MuData object.
    """
    all_X = []
    for m, adat in mdata.mod.items():
        ind_m = mdata.obsm[m]  # Index mapping for the modality
        Xi = adat.X            # Feature matrix for the modality

        if sparse:
            X_pad = sp.csr_matrix((mdata.n_obs, Xi.shape[1]))
        else:
            X_pad = np.full((mdata.n_obs, Xi.shape[1]), fill_value=fillna)

        if sparse and isinstance(Xi, np.ndarray):
            X_pad[ind_m, :] = sp.csr_matrix(Xi)
        elif sparse and sp.issparse(Xi):
            X_pad[ind_m, :] = Xi
        elif not sparse and isinstance(Xi, np.ndarray):
            X_pad[ind_m, :] = Xi
        elif not sparse and sp.isspmatrix(Xi):
            X_pad[ind_m, :] = Xi.toarray()
        else:
            raise ValueError("X must be a ndarray or sparse matrix.")

        all_X.append(X_pad)

    if sparse:
        return sp.hstack(all_X)
    else:
        return np.concatenate(all_X, axis=1)

def convert_mudata_to_anndata(
    mdata: mu.MuData,
    sparse: bool = True,
    fillna: float = 0.0,
    obs: Optional[List[str]] = None,
    obsm: Optional[List[str]] = None,
) -> ad.AnnData:
    """
    Convert a MuData object to an AnnData object.

    Parameters:
        mdata (md.MuData): The MuData object to be converted.
        sparse (bool): Whether to keep the matrix sparse (default: True).
        fillna (float): Value to fill for missing values (default: 0.0).
        obs (Optional[List[str]]): List of `obs` columns to retain in AnnData.
        obsm (Optional[List[str]]): List of `obsm` keys to retain in AnnData.

    Returns:
        ad.AnnData: Converted AnnData object.
    """
    # Extract the feature matrix (X)
    X = get_X_from_mudata(mdata, sparse=sparse, fillna=fillna)

    # Initialize AnnData with the feature matrix
    adata = ad.AnnData(X=X)

    # Transfer obs metadata (rows metadata)
    if obs is not None:
        adata.obs = mdata.obs[obs]

    # Transfer obsm metadata (multi-dimensional rows metadata)
    if obsm is not None:
        for k in obsm:
            adata.obsm[k] = mdata.obsm[k]

    return adata



def run_benchmark(mdata_path, result_dir,result_dir2):
    # 数据读取

    # embedding1_path = os.path.join(result_dir, "final_embeddings_full.csv")
    # if os.path.exists(embedding1_path):
    #     embeddings1 = pd.read_csv(embedding1_path, index_col=0)

    # # 判定并读取 embeddings2
    # embedding2_path = os.path.join(result_dir, "final_embeddings_bipartitle.csv")
    # if os.path.exists(embedding2_path):
    #     embeddings2 = pd.read_csv(embedding2_path, index_col=0)


    # embedding3_path = os.path.join(result_dir, "final_embeddings_matched.csv")
    # if os.path.exists(embedding3_path):
    #     embeddings3 = pd.read_csv(embedding3_path, index_col=0)

    # # # 判定并读取 embeddings4
    # embedding4_path = os.path.join(result_dir2, "final_embeddings_matched_add_feat_1.csv")
    # if os.path.exists(embedding4_path):
    #     embeddings4 = pd.read_csv(embedding4_path, index_col=0)

    embedding5_path = os.path.join(result_dir, "final_embeddings_matched_add_feat.csv")
    if os.path.exists(embedding5_path):
        embeddings5 = pd.read_csv(embedding5_path, index_col=0)

    mdata = mu.read(mdata_path)
    mdata.obs['cell_type'] = mdata.obs['atac:cell_type'].combine_first(mdata.obs['rna:cell_type']).astype("category")
    mdata.obs['batch'] = mdata.obs['atac:batch'].combine_first(mdata.obs['rna:batch']).astype("category")

    # 获取 ATAC 和 RNA 的 obs.index
    atac_index = mdata.mod['atac'].obs.index
    rna_index = mdata.mod['rna'].obs.index

    atac_lsi_pca_df = pd.DataFrame(
        mdata.mod['atac'].obsm['lsi_pca'], index=atac_index
    )
    rna_lsi_pca_df = pd.DataFrame(
        mdata.mod['rna'].obsm['lsi_pca'], index=rna_index
    )

    # 使用 mdata.obs.index 对齐
    obs_index = mdata.obs.index
    atac_lsi_pca_aligned = atac_lsi_pca_df.reindex(obs_index)
    rna_lsi_pca_aligned = rna_lsi_pca_df.reindex(obs_index)

    # 合并：优先使用 ATAC 的值
    lsi_pca_combined = atac_lsi_pca_aligned.combine_first(rna_lsi_pca_aligned)

    # 转回 NumPy 数组
    mdata.obsm["Unintegrated"] = lsi_pca_combined.to_numpy()
    # 检查结果



    #mdata = mdata["rna"]
    print("Data loaded")

    # import ipdb; ipdb.set_trace()
    #sc.tl.pca(mdata, n_comps=30, use_highly_variable=False)
    # mdata.obsm["Unintegrated"] = mdata['rna'].obsm["lsi_pca"]
    # 将对齐后的 embeddings 加入 mdata
    # mdata.obsm["GMI_full"] = embeddings1.values[:17055,]
    # mdata.obsm["GMI_bipartitle"] = embeddings2.values[:17055,]
    # mdata.obsm["GMI_matched"] = embeddings3.values[:17055,]
    # mdata.obsm["GMI_matched_add_feat_1"] = embeddings4.values[:17055,]

    mdata.obsm["GMI_matched+batch_rem"] = embeddings5.values[:76954,]
    # mdata.obs["cell_type"] = mdata.obs["cell_type"]
    # mdata.obsm["Harmony_pca"] = harmonize(
    #     mdata.obsm["lsi_pca"], mdata.obs, batch_key="batch"
    # )
    adata = convert_mudata_to_anndata(
        mdata=mdata,
        sparse=True,
        fillna=0.0,
        obs=['cell_type', 'batch'],          # 指定保留的 obs 列
        obsm=['Unintegrated', 'GMI_matched+batch_rem']  # 指定保留的 obsm 键
    )
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=[
            "Unintegrated",
            # "GMI_full",
            # "GMI_bipartitle",
            # "GMI_matched",
            #"GMI_matched_add_feat_1",
            "GMI_matched+batch_rem",
            # "Harmony_pca",
        ],
        #pre_integrated_embedding_obsm_key="lsi_pca",
        n_jobs=-1,
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False,nmi_ari_cluster_labels_leiden=True,)
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, save_dir=f"{result_dir}")
    
    # 打印详细的结果数据框
    df = bm.get_results(min_max_scale=False)
    df_transposed = df.transpose()
    print(df_transposed)
    df_transposed.to_csv(f"{result_dir}/benchmark_results.csv")


if __name__ == "__main__":
    import os

    result_dir = "/home/yuyipei/graph_mosaic_integration/result"
    mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
    run_benchmark(
        mdata_path,
        result_dir,
        result_dir
    )
