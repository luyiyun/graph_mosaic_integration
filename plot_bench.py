import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# 加载文件
adata = sc.read('/home/yuyipei/graph_mosaic_integration/result/pbmc_comparison.h5ad')
import ipdb; ipdb.set_trace()
# 查看哪些嵌入数据可用
print(adata.obsm.keys())

# 假设选择 'scMoMaT_s0' 作为嵌入数据
embedding = adata.obsm['scMoMaT_s0']

# 查看标签信息（如果存在）
labels = adata.obs['placeholder']  # 使用 'placeholder' 作为标签列

# 绘制散点图
sns.set(style="white", palette="muted")
plt.figure(figsize=(8, 6))

# 使用 scMoMaT_s0 进行绘图
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(label='Label')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Embedding of scMoMaT_s0 with Labels')
plt.show()
