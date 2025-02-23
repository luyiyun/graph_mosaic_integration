# 导入需要的库
import torch
import pandas as pd
import scanpy as sc
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering
import matplotlib.pyplot as plt


import mudata as mu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mdata_path = "/root/autodl-tmp/Human_Lymph_Node/processed/gmi_data.h5mu"
embedding_path = '/root/graph_mosaic_integration/results/alpha_0.04_2025-02-20_09-04/final_embeddings.csv'
embeddings = pd.read_csv(embedding_path, index_col=0)
# 将对齐后的嵌入加入到 mdata
mdata = mu.read(mdata_path)

mdata.obsm["X_embeddings"] = embeddings.loc[mdata.obs.index].to_numpy()
id_to_idx = {cid:i for i,cid in enumerate(mdata.obs.index)}

for mod in mdata.mod:
    # 提取模态索引
    mod_ids = mdata[mod].obs.index
    # 获取匹配索引
    indices = [id_to_idx[cid] for cid in mod_ids]
    # 分配嵌入
    mdata[mod].obsm['feat'] = mdata.obsm["X_embeddings"][indices]



adata_omics1=mdata.mod['rna']
adata_omics2=mdata.mod['adt']

########################################
# 2. 构建邻接图
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
# 3. 模型训练
########################################
print("\n=== Step 6: 模型训练 ===")
model = Train_SpatialGlue(data, datatype='10x', device=device)
output = model.train()

print("\n模型输出键值:", output.keys())
print(f"整合嵌入维度: {output['SpatialGlue'].shape}")

########################################
# 4. 存储结果
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
# 5. 聚类分析
########################################
print("\n=== Step 8: 聚类 ===")
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=6, method='leiden')

print("\n聚类结果分布:")
print(adata.obs['SpatialGlue'].value_counts())

########################################
# 6. 可视化
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
axs[2].hist(adata.obsm['alpha'], bins=30, color=['skyblue', 'orange'])
axs[2].set_title('Alpha Weight Distribution')

plt.tight_layout()
plt.savefig("SpatialGMI_Full_Results.png")
print("\n可视化结果已保存为 SpatialGMI_Full_Results.png")
plt.show()