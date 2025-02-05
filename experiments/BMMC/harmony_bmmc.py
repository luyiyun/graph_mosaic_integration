import os
import numpy as np
import random
import torch
import mudata as mu
from harmony import harmonize
import anndata as ad
from gmi import (
    convert_mudata_to_anndata,
    data_infor_integrate,
)

def setup_environment(seed):
    """
    设置运行环境并指定随机种子。

    参数:
        seed: int, 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_harmony_pipeline(mdata_path, h5ad_path, seeds):
    """
    运行 Harmony 算法并保存结果。

    参数:
        mdata_path: str, MuData 文件路径。
        h5ad_path: str, 保存的 h5ad 文件路径。
        seeds: list, 包含随机种子的列表。
    """
    # 读取 MuData 文件
    mdata = mu.read(mdata_path)
    print(mdata)
    mdata = data_infor_integrate(mdata, feature_key="lsi_pca", saved_feature_name="lsi_pca", target_attr="obsm",dim_limit=100)
    mdata.obsm["Unintegrated"] = mdata.obsm["lsi_pca"]
    mdata = data_infor_integrate(mdata, feature_key="label", saved_feature_name="label", target_attr="obs")
    mdata = data_infor_integrate(mdata, feature_key="batch", saved_feature_name="batch", target_attr="obs")

    # 初始化 AnnData 对象
    adata = convert_mudata_to_anndata(
        mdata=mdata,
        sparse=True,
        fillna=0.0,
        obs=['label', 'batch'],  # 指定保留的 obs 列
        obsm=["Unintegrated"]  # 指定保留的 obsm 键
    )

    for seed in seeds:
        print(f"运行随机种子: {seed}")

        # 设置环境
        setup_environment(seed)

        # 使用 Harmony 校正批次效应
        adata.obsm[f"Harmony_{seed}"] = harmonize(adata.obsm["Unintegrated"], adata.obs, batch_key="batch")

    # 保存结果至 h5ad 文件
    adata.write(h5ad_path)
    print(f"结果已保存到: {h5ad_path}")

# 示例调用
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--mdata_path",
        type=str,
        default="/data/share_data/yuytest/gmi_data/bmmc.h5mu",
        help="MuData 文件路径",
    )
    parser.add_argument(
        "--h5ad_path",
        type=str,
        default="/home/yuyipei/graph_mosaic_integration/result/harmony_bmmc/embeddings.h5ad",
        help="保存的 h5ad 文件路径",
    )
    args = parser.parse_args()

    # 定义随机种子列表
    seeds = [1, 2, 3, 4, 5]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.h5ad_path), exist_ok=True)

    # 运行 Harmony 管道
    run_harmony_pipeline(args.mdata_path, args.h5ad_path, seeds)
