import scipy.io as sio
import numpy as np
import pandas as pd
import scanpy as sc

# 文件路径
rna_counts_path = '/data/share_data/yuytest/gmi_data/unprocessed/uterus/RNA/highvar_counts_rna.mtx'
atac_counts_path = '/data/share_data/yuytest/gmi_data/unprocessed/uterus/ATAC/highvar_counts_atac.mtx'
region2gene_path = '/data/share_data/yuytest/gmi_data/unprocessed/uterus/highvar_region2gene.mtx'
rna_h5ad_path = '/data/share_data/yuytest/gmi_data/unprocessed/uterus/RNA/Uterus_Wang_2020.h5ad'
atac_h5ad_path = '/data/share_data/yuytest/gmi_data/unprocessed/uterus/ATAC/adata_anno.h5ad'

# 读取 RNA 数据
rna_counts = sio.mmread(rna_counts_path).T.tocsr()
print("RNA counts shape:", rna_counts.shape)
print("RNA counts partial data:")
print(rna_counts[:5, :5].toarray())

# 读取 ATAC 数据
atac_counts = sio.mmread(atac_counts_path).T.tocsr()
print("ATAC counts shape:", atac_counts.shape)
print("ATAC counts partial data:")
print(atac_counts[:5, :5].toarray())

# 读取区域-基因关系矩阵
region2gene = sio.mmread(region2gene_path).T.tocsr()
print("Region-to-gene shape:", region2gene.shape)
print("Region-to-gene partial data:")
print(region2gene[:5, :5].toarray())

# 读取 RNA h5ad 数据（只取部分数据）
rna_adata = sc.read_h5ad(rna_h5ad_path, backed='r')
print("RNA adata shape:", rna_adata.shape)
print("RNA adata observations (partial):")
print(rna_adata.obs.iloc[:5])
print("RNA adata variables (partial):")
print(rna_adata.var.iloc[:5])

# 提取部分表达矩阵
rna_partial = rna_adata[:5, :5].X
if hasattr(rna_partial, "toarray"):
    print("RNA counts (partial):")
    print(rna_partial.toarray())
else:
    print("RNA counts (partial):")
    print(rna_partial)

# 读取 ATAC h5ad 数据（只取部分数据）
atac_adata = sc.read_h5ad(atac_h5ad_path, backed='r')
print("ATAC adata shape:", atac_adata.shape)
print("ATAC adata observations (partial):")
print(atac_adata.obs.iloc[:5])
print("ATAC adata variables (partial):")
print(atac_adata.var.iloc[:5])
import ipdb; ipdb.set_trace()
# 提取部分表达矩阵
atac_partial = atac_adata[:5, :5].X
if hasattr(atac_partial, "toarray"):
    print("ATAC counts (partial):")
    print(atac_partial.toarray())
else:
    print("ATAC counts (partial):")
    print(atac_partial)

# 检查标签或注释信息
if 'cell_type' in rna_adata.obs.columns:
    print("RNA cell types:")
    print(rna_adata.obs['cell_type'].value_counts())
else:
    print("No cell_type column in RNA data")

if 'annotation' in atac_adata.obs.columns:
    print("ATAC annotations:")
    print(atac_adata.obs['annotation'].value_counts())
else:
    print("No annotation column in ATAC data")
