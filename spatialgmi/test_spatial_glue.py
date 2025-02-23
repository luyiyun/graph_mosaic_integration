# 导入需要的库
import os
import torch
import pandas as pd
import scanpy as sc
import SpatialGlue
from SpatialGlue.preprocess import fix_seed, clr_normalize_each_cell, pca, construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置数据路径
file_fold = '/root/autodl-tmp/Human_Lymph_Node/'

########################################
# 1. 数据加载
########################################
print("\n=== Step 1: 加载数据 ===")
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')

# 确保变量名唯一（修复加载时的警告）
adata_omics1.var_names_make_unique()  # 添加此行
adata_omics2.var_names_make_unique()  # 添加此行

# 检查初始数据结构
print("\nRNA数据初始结构:")
print(f"Cells: {adata_omics1.n_obs}, Genes: {adata_omics1.n_vars}")  # 修改为 n_vars
print(f"观察字段 (adata_omics1.obs): {adata_omics1.obs.columns.tolist()}")
print(f"特征字段 (adata_omics1.var): {adata_omics1.var.columns.tolist()}")

print("\nADT数据初始结构:")
print(f"Cells: {adata_omics2.n_obs}, Proteins: {adata_omics2.n_vars}")  # 修改为 n_vars
print(f"观察字段 (adata_omics2.obs): {adata_omics2.obs.columns.tolist()}")

########################################
# 2. RNA数据预处理
########################################
print("\n=== Step 2: RNA预处理 ===")
# 基因过滤
sc.pp.filter_genes(adata_omics1, min_cells=10)
print(f"\n过滤后基因数: {adata_omics1.n_vars}")

# 高变基因选择
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
print("\n高变基因统计:")
print(adata_omics1.var['highly_variable'].value_counts())

# 归一化和标准化
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

# 显示处理后的数据分布
print("\n处理后RNA数据统计摘要:")
print(pd.DataFrame(adata_omics1.X).describe())

########################################
# 3. RNA数据降维
########################################
print("\n=== Step 3: RNA PCA降维 ===")
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
# 修改下方 n_comps 参数中的 adata_omics2.n_vars
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)  # 确保此处是 n_vars
print(f"\nPCA后的特征矩阵形状: {adata_omics1.obsm['feat'].shape}")
print("前5个细胞的PCA特征样例:")
print(pd.DataFrame(adata_omics1.obsm['feat']).head())

########################################
# 4. ADT数据预处理
########################################
print("\n=== Step 4: ADT预处理 ===")
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

print("\nCLR归一化后的ADT数据统计:")
print(pd.DataFrame(adata_omics2.X).describe())
print(f"\nADT PCA后的特征矩阵形状: {adata_omics2.obsm['feat'].shape}")

########################################
# 5. 构建邻接图
########################################
print("\n=== Step 5: 构建邻接图 ===")
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype='10x')
# import ipdb;ipdb.set_trace()
# # 显示图结构信息（假设返回的是PyG Data对象）
# print("\n图结构信息:")
# print(f"节点数: {data.num_nodes}")
# print(f"边数: {data.num_edges}")
# print(f"节点特征维度: {data.x.shape}")

########################################
# 6. 模型训练
########################################
print("\n=== Step 6: 模型训练 ===")
model = Train_SpatialGlue(data, datatype='10x', device=device)
output = model.train()

print("\n模型输出键值:", output.keys())
print(f"整合嵌入维度: {output['SpatialGlue'].shape}")

########################################
# 7. 存储结果
########################################
print("\n=== Step 7: 存储结果 ===")
adata = adata_omics1.copy()
adata.obsm.update({
    'emb_latent_omics1': output['emb_latent_omics1'],
    'emb_latent_omics2': output['emb_latent_omics2'],
    'SpatialGlue': output['SpatialGlue'],
    'alpha': output['alpha'],
    'alpha_omics1': output['alpha_omics1'],
    'alpha_omics2': output['alpha_omics2']
})

print("\n最终adata的关键obsm字段:")
print(adata.obsm.keys())

########################################
# 8. 聚类分析
########################################
print("\n=== Step 8: 聚类 ===")
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=6, method='leiden')

print("\n聚类结果分布:")
print(adata.obs['SpatialGlue'].value_counts())

########################################
# 9. 可视化
########################################
print("\n=== Step 9: 可视化 ===")
sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)
sc.tl.umap(adata)

# 创建可视化面板
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# UMAP可视化
sc.pl.umap(adata, color='SpatialGlue', ax=axs[0], title='UMAP Clustering', show=False)

# 空间坐标可视化
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=axs[1], 
                title='Spatial Distribution', s=50, show=False)

# 显示alpha权重分布
axs[2].hist(adata.obsm['alpha'], bins=30, color='skyblue')
axs[2].set_title('Alpha Weight Distribution')

plt.tight_layout()
plt.savefig("SpatialGlue_Full_Results.png")
print("\n可视化结果已保存为 SpatialGlue_Full_Results.png")
plt.show()