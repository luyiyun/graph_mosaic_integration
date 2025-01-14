import os
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import mudata as mu
import scglue
from itertools import chain
from datetime import datetime
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
    adt = mdata.mod['adt']

    # 预处理 RNA 数据
    rna.layers["counts"] = rna.X.copy()  # 保留原始计数数据
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="cell_ranger")
    # sc.pp.normalize_total(rna)
    # sc.pp.log1p(rna)
    sc.pp.scale(rna)

    # 预处理 ATAC 数据
    atac.layers["counts"] = atac.X.copy()  # 保留原始计数数据
    sc.pp.highly_variable_genes(atac, n_top_genes=8000, flavor="cell_ranger")
    # sc.pp.normalize_total(atac)
    # sc.pp.log1p(atac)
    sc.pp.scale(atac)
    if "highly_variable" not in adt.var:
        print("ADT 数据未计算高可变特征，将使用所有特征。")
    else:
        print("ADT 数据已计算高可变特征。")
    # 构建 guidance 图
    guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
    guidance.add_nodes_from(adt.var_names)  # 添加 ADT 的节点
    for node in adt.var_names:
        guidance.add_edge(node, node, weight=1, sign=1)  # 添加 ADT 的自环边
    scglue.graph.check_graph(guidance, [rna, atac, adt])  # 检查图结构

    # 配置数据集
    scglue.models.configure_dataset(
        rna, "Normal", use_highly_variable=True,
        use_layer="counts",
        #use_rep="lsi_pca"  # 使用原始计数数据
    )
    scglue.models.configure_dataset(
        atac, "Normal", use_highly_variable=True,
        use_layer="counts",
        #use_rep="lsi_pca"  # 使用原始计数数据
    )
    scglue.models.configure_dataset(
        adt, "NB",  # 使用负二项分布
        use_highly_variable=False,  # ADT 没有高可变特征，设置为 False
        use_rep="lsi_pca"  # 使用原始数据（没有降维）
    )

    # 构建高可变特征子图
    guidance_hvf = guidance.subgraph(chain(
        rna.var.query("highly_variable").index,
        atac.var.query("highly_variable").index,
        adt.var_names  # 使用 ADT 的所有特征
    )).copy()

    # 训练 SCGLUE 模型
    glue = scglue.models.fit_SCGLUE(
        {"rna": rna, "atac": atac, 'adt': adt}, guidance,
        fit_kws={"directory": os.path.join(result_path, "glue_full")}  
    )

    # 保存模型
    glue.save(os.path.join(result_path, "glue.dill"))

    # 计算 embedding
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    adt.obsm["X_glue"] = glue.encode_data("adt", adt)

    # 合并 embedding
    combined = ad.concat([rna, atac, adt])

    # 计算邻居图
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")

    # 计算 feature embeddings
    feature_embeddings = glue.encode_graph(guidance_hvf)
    feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)

    # 分配 feature embeddings
    rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    adt.varm["X_glue"] = feature_embeddings.reindex(adt.var_names).to_numpy()

    # 保存结果
    
    combined.write(os.path.join(result_path, "glue_bmmc_embedding.h5ad"))
    print(f"随机种子 {random_seed} 的结果已保存到: {result_path}")


# 主程序
if __name__ == "__main__":
    # 参数设置
    mdata_path = "/data/share_data/yuytest/gmi_data/bmmc.h5mu"
    result_dir = "./result/bmmc"
    random_seeds = [1, 2, 3, 4, 5]  # 5 个不同的随机种子

    # 循环训练
    for seed in random_seeds:
        train_scglue(mdata_path, result_dir, seed)