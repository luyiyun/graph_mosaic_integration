import os
import pandas as pd
import anndata as ad
import scanpy as sc
import mudata as mu
import scglue
from itertools import chain
from datetime import datetime
import numpy as np
import random

def set_seed(seed):
    """
    设置随机种子
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_scglue(mdata_path, result_dir, random_seed):
    """
    训练 SCGLUE 模型并保存结果

    参数:
        mdata_path (str): MuData 数据路径
        result_dir (str): 结果保存目录
        random_seed (int): 随机种子
    """
    # 设置随机种子
    set_seed(random_seed)

    # 创建结果目录
    result_path = os.path.join(result_dir, f"seed_{random_seed}")
    os.makedirs(result_path, exist_ok=True)

    # 读取数据
    mdata = mu.read(mdata_path)
    rna = mdata.mod['rna']
    atac = mdata.mod['atac']
    protein = mdata.mod['protein']

    # 预处理 RNA 数据
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="cell_ranger")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)

    # 构建 guidance 图
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    guidance.add_nodes_from(protein.var_names)
    for node in protein.var_names:
        guidance.add_edge(node, node, weight=1, sign=1)
    scglue.graph.check_graph(guidance, [rna, atac, protein])

    # 配置数据集
    scglue.models.configure_dataset(
        rna, "NB", use_highly_variable=True,
        use_layer="counts", use_rep="lsi_pca"
    )
    scglue.models.configure_dataset(
        atac, "NB", use_highly_variable=True,
        use_rep="lsi_pca"
    )
    scglue.models.configure_dataset(
        protein, "Normal",
        use_highly_variable=False,
        use_rep="lsi_pca"
    )

    # 构建高可变特征子图
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index,
        protein.var_names
    )).copy()

    # 训练 SCGLUE 模型
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac, 'protein': protein}, guidance,
        fit_kws={"directory": os.path.join(result_path, "glue_full")}
    )

    # 保存模型
    glue.save(os.path.join(result_path, "glue.dill"))

    # 计算 embedding
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    protein.obsm["X_glue"] = glue.encode_data("protein", protein)

    # 合并 embedding
    combined = ad.concat([rna, atac, protein])

    # 计算邻居图
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")

    # 计算 feature embeddings
    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)

    # 分配 feature embeddings
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    protein.varm["X_glue"] = feature_embeddings.reindex(protein.var_names).to_numpy()

    # 保存结果
    
    combined.write(os.path.join(result_path, "glue_pbmc_embedding.h5ad"))
    print(f"随机种子 {random_seed} 的结果已保存到: {result_path}")


# 主程序
if __name__ == "__main__":
    # 参数设置
    mdata_path = "/data/share_data/yuytest/gmi_data/pbmc.h5mu"
    result_dir = "./result/pbmc"
    random_seeds = [42, 123, 456, 789, 999]  # 5 个不同的随机种子

    # 循环训练
    for seed in random_seeds:
        train_scglue(mdata_path, result_dir, seed)