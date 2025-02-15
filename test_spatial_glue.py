# 导入需要的库
import os
import torch
import pandas as pd
import scanpy as sc
import SpatialGlue
from SpatialGlue.preprocess import fix_seed,clr_normalize_each_cell, pca,construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
from SpatialGlue.utils import clustering

import matplotlib.pyplot as plt

# 设置设备（使用GPU，如果可用；否则使用CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置R环境路径，用于后续与R相关的操作
os.environ['R_HOME'] = '/usr/local/'

# 读取数据：指定数据文件路径
file_fold = '/data/share_data/yuytest/gmi_data/spatial/Human_Lymph_Node/'

# 读取两个不同组学的h5ad文件
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')  # RNA组学数据
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')  # 蛋白质组学数据

# 确保每个数据集的基因名称唯一
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# 指定数据类型为10x
data_type = '10x'

# 固定随机种子，确保实验结果可重复

random_seed = 2022  # 设置随机种子值
fix_seed(random_seed)  # 调用fix_seed函数固定种子

# 从SpatialGlue的预处理模块导入必要的功能：clr归一化和PCA


# 对RNA数据进行预处理
sc.pp.filter_genes(adata_omics1, min_cells=10)  # 过滤掉在少于10个细胞中表达的基因
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)  # 选择3000个高变基因
sc.pp.normalize_total(adata_omics1, target_sum=1e4)  # 归一化处理，确保每个细胞的总表达量为1e4
sc.pp.log1p(adata_omics1)  # 对数据进行log(x+1)转化
sc.pp.scale(adata_omics1)  # 对数据进行标准化

# 提取高变基因后的数据，并进行PCA降维
adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]  # 只保留高变基因
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)  # 使用PCA进行降维，n_comps是维度数

# 对蛋白质组学数据进行CLR归一化
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)  # 对蛋白质数据进行标准化
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)  # 同样进行PCA降维

# 构建邻接图，用于空间组学数据的整合
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

# 定义模型

model = Train_SpatialGlue(data, datatype=data_type, device=device)  # 初始化模型，传入数据和设备

# 训练模型
output = model.train()  # 调用train方法进行模型训练

# 将模型的嵌入结果存储到adata中
adata = adata_omics1.copy()  # 创建adata的副本，以保留原始数据
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()  # 存储RNA数据的嵌入表示
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()  # 存储蛋白质数据的嵌入表示
adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()  # 存储整合后的空间表示
adata.obsm['alpha'] = output['alpha']  # 存储alpha参数
adata.obsm['alpha_omics1'] = output['alpha_omics1']  # 存储RNA数据的alpha参数
adata.obsm['alpha_omics2'] = output['alpha_omics2']  # 存储蛋白质数据的alpha参数

# 进行聚类分析，选择聚类工具（mclust, leiden, louvain），默认是leiden

tool = 'leiden'  # 选择聚类工具
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=6, method=tool, use_pca=True)  # 调用clustering函数进行聚类

# 可视化结果


# 创建一个包含两个子图的图形
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))

# 计算邻居关系并使用UMAP进行降维
sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)  # 使用SpatialGlue嵌入进行邻居计算
sc.tl.umap(adata)  # 进行UMAP降维

# 绘制UMAP图，显示SpatialGlue聚类结果
sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)

# 绘制空间嵌入图，显示SpatialGlue聚类结果
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=25, show=False)

# 调整图形布局，确保子图之间有适当的间距
plt.tight_layout(w_pad=0.3)

# 保存图形到当前目录
plt.savefig("SpatialGlue_Results.png")  # 保存图像为PNG文件，文件名为“SpatialGlue_Results.png”

# 显示图形
plt.show()
