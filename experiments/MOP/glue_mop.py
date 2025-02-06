import os
import pandas as pd
import anndata as ad
import networkx as nx
import scanpy as sc
import mudata as mu
import scglue
from itertools import chain
from scglue.models import SCGLUEModel, PairedSCGLUEModel
from matplotlib import rcParams
from datetime import datetime
from gmi import (
    GraphMosaicIntegration,
    run_benchmark,
    data_infor_integrate,
    plot_umap,
)

# 设置路径
mdata_path = "/data/share_data/yuytest/gmi_data/MOP.h5mu"
result_path = "./result/glue_MOP"
os.makedirs(result_path, exist_ok=True)

# 读取数据
mdata = mu.read(mdata_path)
rna = mdata.mod['rna']
atac = mdata.mod['atac']

# 预处理 RNA 数据
rna.layers["counts"] = rna.X.copy()
sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="cell_ranger")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)

# 构建 guidance graph
guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
scglue.graph.check_graph(guidance, [rna, atac])

# 配置数据集
scglue.models.configure_dataset(
    rna, "NB", use_highly_variable=True,
    use_layer="counts", use_rep="lsi_pca"
)
scglue.models.configure_dataset(
    atac, "NB", use_highly_variable=True,
    use_rep="lsi_pca"
)

# 提取高可变特征的子图
guidance_hvf = guidance.subgraph(chain(
    rna.var.query("highly_variable").index,
    atac.var.query("highly_variable").index
)).copy()

# 循环 5 次，每次使用不同的随机种子
for i in range(1, 6):
    print(f"Running SCGLUE with random seed {i}...")
    
    # 训练 SCGLUE 模型
    glue = scglue.models.fit_SCGLUE(
        adatas={"rna": rna, "atac": atac},
        graph=guidance,
        model=PairedSCGLUEModel,
        init_kws={"random_seed": i}  # 设置随机种子
    )
    # 将编码结果存储到 obsm 中
    rna.obsm[f"X_glue_{i}"] = glue.encode_data("rna", rna)
    atac.obsm[f"X_glue_{i}"] = glue.encode_data("atac", atac)
    
    # 保存模型（可选）
    glue.save(os.path.join(result_path, f"glue_model_seed_{i}.dill"))

# 合并 RNA 和 ATAC 数据
combined = ad.concat([rna, atac])


# 保存结果
combined.write(os.path.join(result_path, "glue_MOP_embedding.h5ad"))

print("All runs completed and results saved.")