# 导入需要的库
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import mudata as mu

# 数据加载和预处理
mdata_path = "/root/autodl-tmp/Human_Lymph_Node/processed/gmi_data.h5mu"
embedding_path = '/root/graph_mosaic_integration/results/alpha_0.04_2025-02-20_09-04/final_embeddings.csv'

# 读取数据和嵌入结果
mdata = mu.read(mdata_path)
embeddings = pd.read_csv(embedding_path, index_col=0)

# 将嵌入结果对齐到mdata对象
mdata.obsm["X_gmi"] = embeddings.loc[mdata.obs.index].to_numpy()

# 创建统一的索引映射
id_to_idx = {cid:i for i,cid in enumerate(mdata.obs.index)}

# 将嵌入分配到各模态
for mod in mdata.mod:
    mod_ids = mdata[mod].obs.index
    indices = [id_to_idx[cid] for cid in mod_ids]
    mdata[mod].obsm['X_gmi'] = mdata.obsm["X_gmi"][indices]

# 使用RNA模态作为基础adata对象
adata = mdata.mod['rna'].copy()

########################################
# 自动聚类分析（与原流程相同方法）
########################################
print("\n=== 自动聚类分析 ===")

# 使用GMI嵌入进行邻居计算
sc.pp.neighbors(adata, use_rep='X_gmi', n_neighbors=10)

# 自动Leiden聚类（保持与原流程相同的参数）
sc.tl.leiden(adata, resolution=0.8, key_added='gmi_clusters')

print("\n聚类结果分布:")
print(adata.obs['gmi_clusters'].value_counts())

########################################
# 可视化（保持与原流程一致的布局）
########################################
print("\n=== 可视化 ===")

# 降维可视化
sc.tl.umap(adata)

# 创建对比可视化面板
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# UMAP可视化
sc.pl.umap(adata, color='gmi_clusters', 
          ax=axs[0], title='GMI Clustering (UMAP)', 
          palette='tab20', show=False)

# 空间分布可视化
sc.pl.embedding(adata, basis='spatial', 
               color='gmi_clusters', ax=axs[1],
               title='Spatial Distribution', 
               s=50, palette='tab20', show=False)

# 嵌入空间可视化
axs[2].scatter(adata.obsm['X_gmi'][:,0], 
              adata.obsm['X_gmi'][:,1],
              c=adata.obs['gmi_clusters'].astype('category').cat.codes,
              cmap='tab20', s=5)
axs[2].set_title('GMI Embedding Space')
axs[2].set_xlabel('Dimension 1')
axs[2].set_ylabel('Dimension 2')

plt.tight_layout()
plt.savefig("GMI_Clustering_Comparison.png", dpi=300)
print("\n可视化结果已保存为 GMI_Clustering_Comparison.png")
plt.show()