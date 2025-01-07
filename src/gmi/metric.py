import os
import numpy as np

import scanpy as sc
import mudata as mu
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation
import matplotlib.pyplot as plt
# from harmony import harmonize
#优化一下复制输入部分
import anndata as ad

import numpy as np
import scipy.sparse as sp
from typing import Optional, List, Union
from harmony import harmonize


def data_infor_integrate(mdata: mu.MuData, feature_key: str,saved_feature_name:str, target_attr:str, dim_limit:int=None):
    """
    mdata 输入的 MuData 对象。
    feature_key要整合的特征键，如 "batch", "coarse_cluster", 或 "lsi_pca"。
    saved_feature_name 整合后的储存name,如 "batch", "label", 或 "Unintegrate"
    target_attr将整合结果存储到 mdata 的属性名称，支持 "obs" 和 "obsm"。
    dim_limit对于多维嵌入（如 lsi_pca），指定最终维度限制，仅适用于 target_attr="obsm"。
    """
    # 提取特征值
    if target_attr == "obs":
        # 提取 obs 中的特征值并对齐
        data = pd.concat(
            [mod.obs[feature_key].reindex(mdata.obs_names)
             for mod in mdata.mod.values() if feature_key in mod.obs],
            axis=1
        )
        # 整合并保存为 category 类型
        mdata.obs[saved_feature_name] = data.bfill(axis=1).iloc[:, 0].astype("category")

    elif target_attr == "obsm":
        # 提取 obsm 中的特征值并对齐
        data = pd.concat(
            [pd.DataFrame(mod.obsm[feature_key], index=mod.obs.index).reindex(mdata.obs.index)
             for mod in mdata.mod.values() if feature_key in mod.obsm],
            axis=1
        )
        # 整合并限制维度
        mdata.obsm[saved_feature_name] = data.bfill(axis=1).iloc[:, :dim_limit].to_numpy()

    else:
        raise ValueError("target_attr must be either 'obs' or 'obsm'.")
    return mdata


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



def run_benchmark(mdata,num_cell, result_dir):
    # 数据读取
    embedding_path = os.path.join(result_dir, "final_embeddings.csv")
    if os.path.exists(embedding_path):
        embeddings = pd.read_csv(embedding_path, index_col=0)

    mdata =mdata
    mdata.obsm["GMI"] = embeddings.values[:num_cell,]
    print("Data loaded")
    adata = convert_mudata_to_anndata(
        mdata=mdata,
        sparse=True,
        fillna=0.0,
        obs=['label', 'batch'],          # 指定保留的 obs 列
        obsm=['GMI']#,"Unintegrated"]  # 指定保留的 obsm 键
    )
    #adata.obsm["Harmony"] = harmonize(adata.obsm["Unintegrated"], adata.obs, batch_key="batch")
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="label",
        embedding_obsm_keys=[
            # "Unintegrated",
            # "GMI_full",
            # "GMI_bipartitle",
            # "GMI_matched",
            #"GMI_matched_add_feat_1",
            "GMI",
            #"Harmony",
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
    print('benchmark result saved')


if __name__ == "__main__":
    import os

    result_dir = "/home/yuyipei/graph_mosaic_integration/result"
    mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
    run_benchmark(
        mdata_path,
        result_dir,
        result_dir
    )
