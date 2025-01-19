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
import itertools
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
    result_path = os.path.join(result_dir, f"seed_{random_seed}")
    os.makedirs(result_path, exist_ok=True)
    

    # 读取数据
    mdata = mu.read(mdata_path)
    rna = mdata.mod['rna']
    atac = mdata.mod['atac']
    met = mdata.mod['met'] 

    # 预处理 RNA 数据
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="cell_ranger")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)

    if "highly_variable" not in met.var:
        print("met 数据未计算高可变特征，将使用所有特征。")
    else:
        print("met 数据已计算高可变特征。")

    # 构建 guidance 图
    guidance = scglue.genomics.window_graph(
        scglue.genomics.Bed(rna.var.assign(name=rna.var_names)).expand(2e3, 0),
        scglue.genomics.Bed(atac.var.assign(name=atac.var_names)),
        window_size=0, attr_fn=lambda l, r, d: {"weight": 1.0, "sign": 1}
    )

    # %%
    for i in met.var_names:
        if i.endswith("_mCH"):
            j = i.replace("_mCH", "")
        elif i.endswith("_mCG"):
            j = i.replace("_mCG", "")
        else:
            raise ValueError("Unexpected var name!")
        guidance.add_edge(j, i, weight=1.0, sign=-1)

    guidance_hvf = scglue.graph.reachable_vertices(guidance, rna.var.query("highly_variable").index)

    # %%
    met.var["highly_variable"] = [item in guidance_hvf for item in met.var_names]
    met.var["highly_variable"].sum()

    # %%
    atac.var["highly_variable"] = [item in guidance_hvf for item in atac.var_names]
    atac.var["highly_variable"].sum()
    
    guidance = scglue.graph.compose_multigraph(guidance, guidance.reverse())
    for i in itertools.chain(rna.var_names, met.var_names, atac.var_names):
        guidance.add_edge(i, i, weight=1.0, sign=1)

    # %%
    subgraph = guidance.subgraph(guidance_hvf)


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
        met, "Normal",  # met 数据是连续值，使用正态分布
        use_highly_variable=False,  #  met 没有高可变特征，设置为 False
        use_rep="lsi_pca"  # 使用 LSI 降维表示
    )

    # # 构建高可变特征子图
    # guidance_hvf = guidance.subgraph(chain(
    #     rna.var.query("highly_variable").index,
    #     atac.var.query("highly_variable").index,
    #     met.var_names  # 使用 met 的所有特征
    # )).copy()

    # 训练 SCGLUE 模型
    # 添加随机种子
    glue = scglue.models.fit_SCGLUE(
        adatas={"rna": rna, "atac": atac, "met": met},
        graph=guidance,
        model=SCGLUEModel,
        init_kws={"random_seed": random_seed}  # 设置随机种子
    )
    # glue = scglue.models.fit_SCGLUE(
    #     {"rna": rna, "atac": atac, 'met': met}, guidance, 
    #     fit_kws={"directory": os.path.join(result_path, "glue_full")},
    #     random_seed=random_seed
    # )

    # 保存模型
    glue.save(os.path.join(result_path, "glue.dill"))

    # 计算 embedding
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    met.obsm["X_glue"] = glue.encode_data("met", met) 

    # 合并 embedding
    combined = ad.concat([rna, atac, met]) 

    # 计算邻居图
    sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")

    # # 计算 feature embeddings
    # feature_embeddings = glue.encode_graph(guidance_hvf)
    # feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)

    # # 分配 feature embeddings
    # rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
    # atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
    # met.varm["X_glue"] = feature_embeddings.reindex(met.var_names).to_numpy()  

    # 保存结果
    
    combined.write(os.path.join(result_path, "glue_trip_embedding.h5ad"))
    print(f"随机种子 {random_seed} 的结果已保存到: {result_path}")


# 主程序
if __name__ == "__main__":
    # 参数设置
    mdata_path = "/data/share_data/yuytest/gmi_data/triple.h5mu"
    result_dir = "./result/triple_glue"
    random_seeds = [1, 2, 3, 4, 5]  # 5 个不同的随机种子

    # 循环训练
    for seed in random_seeds:
        train_scglue(mdata_path, result_dir, seed)