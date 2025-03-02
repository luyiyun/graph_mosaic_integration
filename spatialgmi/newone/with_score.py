import os
import numpy as np
import json
from datetime import datetime
import mudata as mu
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from gmi import (
    GraphMosaicIntegration,
    run_benchmark,
    data_infor_integrate,
    plot_umap,
)
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

from sklearn.metrics import homogeneity_score, mutual_info_score, v_measure_score
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score

# 配置参数

def jaccard_similarity(true_labels, predicted_labels):
    """
    计算Jaccard相似度
    """
    return jaccard_score(true_labels, predicted_labels, average='macro')


def compute_morans_i(feature_values, spatial_coords):
    """
    计算Moran's I空间自相关得分
    :param feature_values: (n_cells, n_features) 的特征矩阵
    :param spatial_coords: (n_cells, 2) 的空间坐标矩阵（假设是二维坐标）
    :return: 每个特征的 Moran's I 得分
    """
    n_cells, n_features = feature_values.shape
    morans_i_scores = []

    # 计算空间距离矩阵
    dist_matrix = np.linalg.norm(spatial_coords[:, np.newaxis] - spatial_coords, axis=2)
    
    # 构建空间权重矩阵（使用高斯核）
    sigma = np.median(dist_matrix)  # 使用中位数作为sigma
    w = np.exp(-dist_matrix / sigma)  # 使用高斯核作为权重
    np.fill_diagonal(w, 0)  # 对角线置零，表示不考虑自己与自己的关系
    w /= np.sum(w, axis=1)[:, None]  # 归一化每行

    # 对每个特征计算 Moran's I
    for i in range(n_features):
        feature_values_centered = feature_values[:, i] - np.mean(feature_values[:, i])
        
        # 计算 Moran's I
        num = np.sum(w * (feature_values_centered[:, None] * feature_values_centered[None, :]))
        denom = np.sum(feature_values_centered ** 2)

        morans_i_scores.append(num / denom)
    
    return np.array(morans_i_scores)

def run_gmi_pipeline(timestamp):
    """GMI整合分析主流程"""
    
    # 1. 数据加载
    try:
        print(f"→ 加载数据: {CONFIG['data_path']}")
        mdata = mu.read(CONFIG['data_path'])
        print(f"数据摘要:\n{mdata}")
    except Exception as e:
        print(f"× 数据加载失败: {str(e)}")
        return

    # 2. 数据信息整合
    try:
        mdata = data_infor_integrate(
            mdata,
            feature_key="batch",
            saved_feature_name="batch",
            target_attr="obs",
        )

        mdata['rna'].obs['x'] = mdata['rna'].obsm['spatial'][:, 0]
        mdata['rna'].obs['y'] = mdata['rna'].obsm['spatial'][:, 1]
        mdata['adt'].obs['x'] = mdata['adt'].obsm['spatial'][:, 0]
        mdata['adt'].obs['y'] = mdata['adt'].obsm['spatial'][:, 1]

        # Batch编码映射
        batch_mapping = {cat: idx + 1 for idx, cat in enumerate(mdata.obs['batch'].cat.categories)}
        mdata.obs['batch'] = mdata.obs['batch'].map(batch_mapping)
    except KeyError as e:
        print(f"× 元数据字段缺失: {str(e)}")
        return
    print(mdata)

    # 3. 模型训练
    # 创建结果目录
    timestamp = timestamp
    result_dir = '/root/graph_mosaic_integration/spatialgmi/newone/result'
    os.makedirs(result_dir, exist_ok=True)
    print(f"RNA数据的大小: {mdata['rna'].shape}")
    print(f"ADT数据的大小: {mdata['adt'].shape}")
    print(f"net的维度: {mdata.varp['net'].shape}")
    print(f"Batch信息: {mdata.obs['batch'].value_counts()}")

    # 初始化模型
    print(CONFIG['model_params'])
    gmi_model = GraphMosaicIntegration(
        num_neg_per_pos=CONFIG['model_params']['num_neg_per_pos'],
        label_smoothing=CONFIG['model_params']['label_smoothing'],
        w_grad_rev=CONFIG['model_params']['alpha'],
        w_loss_cls=CONFIG['model_params']['loss_alpha'],
        #adversarial_training=True,
        adversarial_batching_method="divide",
        val_split=0.1,
        patience=5,
        num_epochs=CONFIG['model_params']['num_epochs'],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        adversartial_balance_weights=CONFIG['model_params']['adversarial_balance_weights'],
        num_epochs_with_balanced_weights=20,
        learning_rate=CONFIG['model_params']['learning_rate'],
        add_batch_embedding=True,
        bilinear=False,
        use_spatial_distance= True,
        distance_threshold= CONFIG['model_params']["distance_threshold"],
    )
    
    # 模型训练
    print(f"\n→ 开始训练 (alpha={alpha})")
    gmi_model.fit(
        mdata, 
        batch_key="batch",
        spatial_keys=["x", "y"],
        spatial_threshold=40,
        spatial_sigma=CONFIG['model_params']['sigma'],
        spatial_alpha=CONFIG['model_params']['w_sigma'],
        feature_interaction_key="net"  # 确保已构建varp['net']
    )
    
    # 保存结果
    gmi_model.save(result_dir)
    print(f"✓ 结果保存至: {result_dir}")
    
    # 保存配置信息
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=2)


# 执行主流程
if __name__ == "__main__":
    # 检查GPU可用性
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f"可用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    for alpha in [0.5,0.6,0.65,0.7,0.8]:
        CONFIG = {
            "data_path": "/root/autodl-tmp/Human_Lymph_Node/processed/rna_adt.h5mu",
            "result_root": "./results",
            "model_params": {
                "num_neg_per_pos": 4,
                "label_smoothing": 0.1,
                "loss_alpha": 0.1,
                "alpha":0.1,
                "neg_sampling_mode": "matched",
                "adversarial_balance_weights": False,
                "num_epochs": 100,
                "learning_rate": 0.05,
                "distance_threshold": 5,
                "sigma": alpha,
                "w_sigma":1,
                "use_spatial_distance": True
            }
        }
        # 执行GMI整合分析主流程
        run_gmi_pipeline(timestamp)

        # 读取保存的嵌入结果并进行聚类与可视化
        print("\n=== 自动聚类分析 ===")
        mdata = mu.read(CONFIG['data_path'])
        embedding_path = '/root/graph_mosaic_integration/spatialgmi/newone/result/final_embeddings.csv'

        embeddings = pd.read_csv(embedding_path, index_col=0)
        annotation_path="/root/autodl-tmp/annotation.csv"
        annotation=pd.read_csv(annotation_path, index_col=0)
        mdata.obsm["X_gmi"] = embeddings.loc[mdata.obs.index].to_numpy()
        # adata = mdata.mod['rna'].copy()
        # adata.obsm['X_gmi']=mdata.obsm['X_gmi']
        # celltype_to_structure = {
        #     'CD34+ SC': 'capsule',          # CD34+ 基质细胞 -> 被膜
        #     'Ccl19lo TRC': 'cortex',        # Ccl19lo TRC -> 皮质
        #     'Cxcl9+ TRC': 'cortex',         # Cxcl9+ TRC -> 皮质
        #     'FDC': 'follicle',              # 滤泡树突状细胞 -> 滤泡
        #     'Inmt+ SC': 'medulla cords',    # Inmt+ SC -> 髓索
        #     'MRC': 'medulla cords',         # 髓质网状细胞 -> 髓索
        #     'Nr4a1+ SC': 'cortex',          # Nr4a1+ SC -> 皮质
        #     'PvC': 'medulla vessels',       # 血管周细胞 -> 髓质血管
        #     'TRC': 'cortex',                # TRC -> 皮质
        # }
        # cc_label_after = adata.obs['CC label'][3484:]
        # cc_label_after_mapped = cc_label_after.map(celltype_to_structure)
        # adata.obs['cell_type'] = np.nan
        # adata.obs.iloc[:3484, adata.obs.columns.get_loc('cell_type')] = annotation.values[:,0]
        # adata.obs.iloc[3484:, adata.obs.columns.get_loc('cell_type')] = cc_label_after_mapped.values
        # print(adata.obs['cell_type'].value_counts())
        #     # 创建左右两个子图
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # # 左侧：按 batch 着色
        # sc.pl.embedding(
        #     adata,
        #     basis="gmi",  
        #     color="batch",
        #     title="GMI by Batch",
        #     legend_loc="on data",  
        #     frameon=False,
        #     ax=ax1,
        #     show=False  
        # )

        # # 右侧：按 cell_type 着色
        # sc.pl.embedding(
        #     adata,
        #     basis="gmi", 
        #     color="cell_type",
        #     title="GMI by Cell Type",
        #     legend_loc="right margin", 
        #     frameon=False,
        #     ax=ax2,
        #     show=False
        # )
        # plt.savefig("/root/graph_mosaic_integration/spatialgmi/newone/umap/batch/umap_batch_vs_cell_type.png", dpi=300, bbox_inches="tight")
        # print('batch and cell type ploted')



        mdata = mdata[:3484]

        print(mdata)
        # 创建统一的索引映射
        id_to_idx = {cid:i for i, cid in enumerate(mdata.obs.index)}

        # 分配嵌入到各模态
        for mod in mdata.mod:
            mod_ids = mdata[mod].obs.index
            indices = [id_to_idx[cid] for cid in mod_ids]
            mdata[mod].obsm['X_gmi'] = mdata.obsm["X_gmi"][indices]
        # 使用RNA模态作为基础adata对象
        adata = mdata.mod['rna'].copy()
        adata2 = sc.read_h5ad("/root/autodl-tmp/adata_all_human_lymph_node_A1.h5ad")
        index_match = adata.obs.index.equals(adata2.obs.index)

        if index_match:
            print("adata 和 adata2 的 obs 索引完全一致。")
        else:
            print("adata 和 adata2 的 obs 索引不一致。")

        # 进行邻居计算
        sc.pp.neighbors(adata, use_rep='X_gmi')

        # 聚类
        sc.tl.leiden(adata, resolution=0.8, key_added='gmi_clusters')
        adata.obs['ground_truth'] = annotation.values[:,0]

        print("\n聚类结果分布:")
        print(adata.obs['gmi_clusters'].value_counts())

        # 可视化
        sc.tl.umap(adata)
        fig, axs = plt.subplots(2, 4, figsize=(22, 10))
        axs = axs.flatten()
        # UMAP可视化
        sc.pl.umap(adata, color='ground_truth', ax=axs[0], title='GMI Clustering (UMAP)', palette='tab20', show=False)

        # 空间分布可视化
        sc.pl.embedding(adata, basis='spatial', color='gmi_clusters', ax=axs[1], title='Spatial Distribution', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata, basis='spatial', color='ground_truth', ax=axs[2], title='ground_truth', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata2, basis='spatial', color='SpatialGlue', ax=axs[3], title='SpatialGlue', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata2, basis='spatial', color='MOFA', ax=axs[4], title='MOFA', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata2, basis='spatial', color='Seurat', ax=axs[5], title='Seurat', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata2, basis='spatial', color='StabMap', ax=axs[6], title='StabMap', s=50, palette='tab20', show=False)
        sc.pl.embedding(adata2, basis='spatial', color='MultiVI', ax=axs[7], title='MultiVI', s=50, palette='tab20', show=False)
        plt.tight_layout()
        umap_dir = f"/root/graph_mosaic_integration/spatialgmi/newone/umap/{alpha}"
        os.makedirs(umap_dir, exist_ok=True)
        plt.savefig(os.path.join(umap_dir, f"umap_result_{alpha}.png"), dpi=300)
        print(f"\n可视化结果已保存为: {os.path.join(umap_dir, f'umap_result.png')}")
        # 获取空间坐标（
        spatial_coords = adata2.obsm['spatial']  # 空间坐标

        # 存储不同方法的 Moran's I 得分
        moran_scores_dict = {}
        # 计算每个方法的 Moran's I 得分
        methods = ['X_gmi','X_StabMap', 'emb', 'origi_pca']
        max_length = 0
        for method in methods:
            if method == 'X_gmi':
                feature_values = adata.obsm['X_gmi']  # 对应于GMI的嵌入结果
            else:
                feature_values = adata2.obsm[method]  # 其他方法的嵌入结果
            moran_scores = compute_morans_i(feature_values, spatial_coords)
            moran_scores_dict[method] = moran_scores
            max_length = max(max_length, len(moran_scores))
        # 填充较小维度的数据，使它们与最大的维度对齐
        for method in methods:
            # 计算每个方法 Moran's I 的维度
            moran_scores = moran_scores_dict[method]
            # 如果维度小于最大维度，进行填充
            if len(moran_scores) < max_length:
                # 使用 NaN 填充
                moran_scores_dict[method] = np.pad(moran_scores, (0, max_length - len(moran_scores)), mode='constant', constant_values=np.nan)

        # 将结果转化为 DataFrame 便于绘图
        moran_scores_df = pd.DataFrame(moran_scores_dict)

        # 将每个方法的 Moran's I 得分整理成长格式，以便于绘图
        moran_scores_df_melted = moran_scores_df.melt(var_name='Method', value_name="Moran's I")

        palette = sns.color_palette("Set2", len(methods))  # 选择一个合适的调色板，'Set2'为常用的颜色调色板

        # 绘制箱式图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y="Moran's I", data=moran_scores_df_melted, palette=palette)
        plt.title('Moran\'s I Scores for Different Methods')
        plt.xlabel('Methods')
        plt.ylabel('Moran\'s I Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(umap_dir, f"umap_result_Moran.png"), dpi=300)
        print(f"\n可视化结果已保存为: {os.path.join(umap_dir, f'umap_result_{alpha}.png')}")
        plt.show()
        # 获取GMI聚类和ground_truth标签

        # 合并真实标签和预测标签
        true_labels = adata.obs['ground_truth'].values
        predicted_labels = adata.obs['gmi_clusters'].values    
        
        # 计算每个指标
        homogeneity = homogeneity_score(true_labels, predicted_labels)
        mutual_info = mutual_info_score(true_labels, predicted_labels)
        v_measure = v_measure_score(true_labels, predicted_labels)
        ami = adjusted_mutual_info_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)

        # 打印每个指标
        print(f"Homogeneity: {homogeneity}")
        print(f"Mutual Information: {mutual_info}")
        print(f"V-Measure: {v_measure}")
        print(f"Adjusted Mutual Information (AMI): {ami}")
        print(f"Normalized Mutual Information (NMI): {nmi}")
        print(f"Adjusted Rand Index (ARI): {ari}")

        evaluation_data_path = '/root/autodl-tmp/evaluation_single_resolution_A1.csv'
        evaluation_data = pd.read_csv(evaluation_data_path)
        x_gmi_results = {
            'Metric': ['Homogeneity', 'Mutual Information', 'V-Measure', 'AMI', 'NMI', 'ARI'],
            'Score': [homogeneity, mutual_info, v_measure, ami, nmi, ari]
        }
        x_gmi_df = pd.DataFrame(x_gmi_results)
        # 直接将X_gmi的行数据添加到evaluation_data的末尾
        x_gmi_row = ['X_gmi'] + x_gmi_df['Score'].tolist()

        # 直接用位置索引添加新行，不依赖列名
        evaluation_data.loc[len(evaluation_data)] = ['X_gmi'] + x_gmi_row[1:]
        evaluation_data.rename(columns={'Unnamed: 0': 'Method'}, inplace=True)
        # 将结果转化为长格式
        evaluation_data_melted = evaluation_data.melt(id_vars=['Method'], value_vars=['homogeneity', 'mutual_info', 'v_measure', 'AMI', 'NMI', 'ARI'],
                                                    var_name='Metric', value_name='Score')
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Score', hue='Method', data=evaluation_data_melted)
        plt.title('Comparison of Metrics for Different Methods')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(umap_dir, f"umap_result_all_score.png"), dpi=300)
        # from scipy.cluster.hierarchy import linkage, dendrogram

        # # 计算数据的相关性矩阵
        # corr_matrix = np.corrcoef(mdata.obsm["X_gmi"], rowvar=False)

        # # 使用层次聚类方法
        # linkage_matrix = linkage(corr_matrix, method='ward')

        # # 绘制聚类热图
        # plt.figure(figsize=(10, 8))
        # step = 5
        # # step2 = 5
        # sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, fmt='.2f', linewidths=.5,
        #             xticklabels=[f"Dim {i}" if i % step == 0 else "" for i in range(1, corr_matrix.shape[0] + 1)],
        #             yticklabels=[f"Dim {i}" if i % step == 0 else "" for i in range(1, corr_matrix.shape[0] + 1)])

        # plt.title('Clustered Heatmap')
        # plt.savefig(os.path.join(umap_dir, f"umap_result_Clustered_Heatmap.png"), dpi=300)
        
        # # 假设你已经进行了聚类，并且将聚类结果保存在obs中
        # # 计算每个簇的DEG，这里以RNA数据为例
        # sc.tl.rank_genes_groups(adata, groupby='gmi_clusters', method='t-test')
        # # 提取DEG的结果
        # degs = adata.uns['rank_genes_groups']['names']

        # # 选取前几个显著的DEG
        # top_degs = degs[:20].tolist()  # 例如，选择前20个显著基因
        # top_degs = [gene for row in top_degs for gene in row]
        # matched_indices = [i for i, gene in enumerate(adata.var_names) if gene in top_degs]
        # # 提取这些基因在RNA表达矩阵中的值
        # degs_expression = adata.X[:, matched_indices].toarray()

        # # 绘制热力图
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(degs_expression, cmap="coolwarm", annot=True, xticklabels=top_degs, yticklabels=adata.obs['gmi_clusters'])
        # plt.title("Heat map of DEGs for each cluster")
        # plt.xlabel('Top DEGs')
        # plt.ylabel('Clusters')
        # plt.savefig(os.path.join(umap_dir, f"umap_result_Clustered_deg_heatmap.png"), dpi=300)

    print("整合分析完成")
