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
from scglue.models import SCGLUEModel, PairedSCGLUEModel

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
    os.makedirs(result_dir, exist_ok=True)

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

    # 训练 SCGLUE 模型
    glue = scglue.models.fit_SCGLUE(
        adatas={"rna": rna, "atac": atac, 'protein': protein},
        graph=guidance,
        model=PairedSCGLUEModel,
        init_kws={"random_seed": random_seed}  # 设置随机种子
    )

    # 计算 embedding
    rna.obsm[f"X_glue_{random_seed}"] = glue.encode_data("rna", rna)
    atac.obsm[f"X_glue_{random_seed}"] = glue.encode_data("atac", atac)
    protein.obsm[f"X_glue_{random_seed}"] = glue.encode_data("protein", protein)

    # 合并数据
    combined = ad.concat([rna, atac])

    # 保存模型（可选）
    glue.save(os.path.join(result_dir, f"glue_model_seed_{random_seed}.dill"))

    print(f"随机种子 {random_seed} 的结果已保存到: {result_dir}")
    return combined


# 主程序
if __name__ == "__main__":
    # 参数设置
    mdata_path = "/data/share_data/yuytest/gmi_data/pbmc.h5mu"
    result_dir = "./result/pbmc_glue"
    random_seeds = [42, 123, 456, 789, 999]  # 5 个不同的随机种子

    # 初始化一个空的 AnnData 对象
    combined = None

    # 循环训练
    for seed in random_seeds:
        result = train_scglue(mdata_path, result_dir, seed)
        if combined is None:
            combined = result  # 第一次运行时初始化 combined
        else:
            # 将新的 embedding 添加到 combined 的 obsm 中
            for key, value in result.obsm.items():
                combined.obsm[key] = value

    # 保存最终结果
    combined.write(os.path.join(result_dir, "glue_pbmc_embedding.h5ad"))
    print(f"所有随机种子的结果已保存到: {os.path.join(result_dir, 'glue_pbmc_embedding.h5ad')}")