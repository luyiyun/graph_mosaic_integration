import os
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.sparse as sp

# 文件路径设置
base_dir = "/data/share_data/yuytest/gmi_data/unprocessed"
rna_path = os.path.join(base_dir, "GSM3681518_MNC_RNA_counts.tsv")
adt_path = os.path.join(base_dir, "GSM3681519_MNC_ADT_counts.tsv")
hto_path = os.path.join(base_dir, "GSM3681520_MNC_HTO_counts.tsv")
output_dir = os.path.join(base_dir, "processed")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# RNA 数据加载和质量控制
rna_counts = pd.read_csv(rna_path, sep="\t", index_col=0)
adata_rna = sc.AnnData(sp.csr_matrix(rna_counts.T))
adata_rna.var_names = rna_counts.index
adata_rna.obs_names = rna_counts.columns

# 添加质控指标
adata_rna.var_names_make_unique()
adata_rna.obs['n_genes'] = (adata_rna.X > 0).sum(1)
adata_rna.obs['n_counts'] = adata_rna.X.sum(1)
adata_rna.obs['percent_mt'] = np.sum(adata_rna[:, adata_rna.var_names.str.startswith('MT-')].X, axis=1) / adata_rna.obs['n_counts'] * 100

# 筛选条件
sc.pp.filter_cells(adata_rna, min_genes=350)
adata_rna = adata_rna[(adata_rna.obs['n_genes'] < 6000) & (adata_rna.obs['n_counts'] < 40000) & (adata_rna.obs['percent_mt'] < 20)]

# ADT 数据加载和质量控制
adt_counts = pd.read_csv(adt_path, sep="\t", index_col=0)
adt_counts = adt_counts.T[adata_rna.obs_names]  # 确保细胞一致
adata_adt = sc.AnnData(sp.csr_matrix(adt_counts))
adata_adt.var_names = adt_counts.index
adata_adt.obs_names = adt_counts.columns
adata_adt.obs['n_counts'] = adata_adt.X.sum(1)

# 筛选条件
adata_adt = adata_adt[(adata_adt.obs['n_counts'] > 500) & (adata_adt.obs['n_counts'] < 15000)]

# HTO 数据加载和处理
hto_counts = pd.read_csv(hto_path, sep="\t", index_col=0)
hto_counts = hto_counts.T[adata_rna.obs_names]  # 确保细胞一致
adata_hto = sc.AnnData(sp.csr_matrix(hto_counts))
adata_hto.var_names = hto_counts.index
adata_hto.obs_names = hto_counts.columns

# 归一化处理和分类
adata_hto.X = np.log1p(adata_hto.X)
hto_thresholds = np.percentile(adata_hto.X, 99, axis=0)
labels = []
for i, cell_counts in enumerate(adata_hto.X):
    is_doublet = np.sum(cell_counts > hto_thresholds) > 1
    labels.append('Doublet' if is_doublet else 'Singlet')
adata_hto.obs['classification'] = labels

# 仅保留 Singlet
adata_hto = adata_hto[adata_hto.obs['classification'] == 'Singlet']

# 交集细胞筛选
intersect_cells = list(set(adata_rna.obs_names) & set(adata_adt.obs_names) & set(adata_hto.obs_names))
adata_rna = adata_rna[intersect_cells]
adata_adt = adata_adt[intersect_cells]

# 保存处理后的数据
sc.write(os.path.join(output_dir, 'rna_processed.h5ad'), adata_rna)
sc.write(os.path.join(output_dir, 'adt_processed.h5ad'), adata_adt)