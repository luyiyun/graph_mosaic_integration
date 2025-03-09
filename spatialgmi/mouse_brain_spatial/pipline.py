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
import squidpy as sq

from sklearn.metrics import homogeneity_score, mutual_info_score, v_measure_score
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
# 配置参数


umap_dir = f"/root/graph_mosaic_integration/spatialgmi/mouse_brain_spatial/result/umap/0.7"
os.makedirs(umap_dir, exist_ok=True)
# 定义函数：绘制空间分布图
def plot_spatial(adata,basis, color, title, save_path=None):
    """绘制空间分布图"""
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        title=title,
        ax=ax,
        show=False,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

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
        mdata = data_infor_integrate(
            mdata,
            feature_key="cell_type",
            saved_feature_name="cell_type",
            target_attr="obs",
        )
       
        mdata['rna'].obs['x'] = mdata['rna'].obsm['spatial'][:, 0]
        mdata['rna'].obs['y'] = mdata['rna'].obsm['spatial'][:, 1]

        # Batch编码映射
        batch_mapping = {cat: idx + 1 for idx, cat in enumerate(mdata.obs['batch'].cat.categories)}
        mdata.obs['batch'] = mdata.obs['batch'].map(batch_mapping)
    except KeyError as e:
        print(f"× 元数据字段缺失: {str(e)}")
        return

    # 3. 模型训练
    # 创建结果目录
    timestamp = timestamp
    result_dir = '/root/graph_mosaic_integration/spatialgmi/mouse_brain_spatial/result'
    os.makedirs(result_dir, exist_ok=True)
    print(f"RNA数据的大小: {mdata['rna'].shape}")
    print(f"ADT数据的大小: {mdata['atac'].shape}")
    print(f"net的维度: {mdata.varp['net'].shape}")
    print(f"Batch信息: {mdata.obs['batch'].value_counts()}")

    plot_spatial(mdata.mod['rna'][mdata.mod['rna'].obs['batch'] == '1'],basis='spatial', color='cell_type', title='Batch 1: Cell Type', save_path=os.path.join(umap_dir, 'batch1_cell_type_initial.png'))
    plot_spatial(mdata.mod['rna'][mdata.mod['rna'].obs['batch'] == '2'],basis='spatial', color='cell_type', title='Batch 2: Cell Type', save_path=os.path.join(umap_dir, 'batch2_cell_type_initial.png'))
    # 初始化模型
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
    
    gmi_model.save(result_dir)
    print(f"✓ 结果保存至: {result_dir}")
    
    # 保存配置信息
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f, indent=2)

# 执行主流程
if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f"可用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    for alpha in [0.7]:
        CONFIG = {
            "data_path": "/root/autodl-tmp/new_data/integrated_data.h5mu",
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
        embedding_path = '/root/graph_mosaic_integration/spatialgmi/mouse_brain_spatial/result/final_embeddings.csv'

        embeddings = pd.read_csv(embedding_path, index_col=0)

        mdata.obsm["X_gmi"] = embeddings.loc[mdata.obs.index].to_numpy()
        mdata = data_infor_integrate(
            mdata,
            feature_key="batch",
            saved_feature_name="batch",
            target_attr="obs",
        )
        mdata = data_infor_integrate(
            mdata,
            feature_key="cell_type",
            saved_feature_name="cell_type",
            target_attr="obs",
        )
        adata = sc.AnnData(mdata.obsm['X_gmi'])
        adata.obs.index = mdata.obs.index
        adata.obs['batch'] = mdata.obs['batch']
        adata.obs['cell_type'] = mdata.obs['cell_type'] 
        adata.obs['mod'] = ['rna'] * 9026 + ['atac'] * (adata.n_obs - 9026)
        # 添加 obs 信息（batch 和 cell_type）
        sc.pp.neighbors(adata)  # 计算邻居图
        sc.tl.umap(adata)  


        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 6))

        sc.pl.umap(
            adata,
            color='mod',
            title='UMAP (colored by mod)',
            ax=ax3,
            show=False
        )

        # 用 batch 染色
        sc.pl.umap(
            adata,
            color='batch',
            title='UMAP (colored by batch)',
            ax=ax1,
            show=False
        )

        # 用 cell_type 染色
        sc.pl.umap(
            adata,
            color='cell_type',
            title='UMAP (colored by cell_type)',
            ax=ax2,
            show=False
        )
        # 调整布局并显示
        plt.tight_layout()

        plt.savefig(os.path.join(umap_dir, f"umap_result_{alpha}.png"), dpi=300)
        plt.show()

        from sklearn.cluster import KMeans

        # 提取 rna 数据
        
        rna_obs_names = mdata.mod['rna'].obs_names
        rna_indices = mdata.obs.index.isin(rna_obs_names)  # 获取 rna 的观测索引
        rna_x_gmi = mdata.obsm['X_gmi'][rna_indices]
        rna = mdata.mod['rna']
        rna.obsm['X_gmi'] = rna_x_gmi  
        # 提取 batch=2 和 batch=3 的数据
        batch2_data = rna[rna.obs['batch'] == '2'].copy()
        batch1_data = rna[rna.obs['batch'] == '1'].copy()

        # 定义函数：基于 X_gmi 进行聚类
        def cluster_with_xgmi(adata, n_clusters=3):
            """使用 X_gmi 进行 KMeans 聚类"""
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            adata.obs['xgmi_cluster'] = kmeans.fit_predict(adata.obsm['X_gmi'])
            adata.obs['xgmi_cluster'] = adata.obs['xgmi_cluster'].astype('category')
            return adata
        # 对 batch=2 和 batch=3 进行聚类
        batch2_data = cluster_with_xgmi(batch2_data, n_clusters=3)
        batch1_data = cluster_with_xgmi(batch1_data, n_clusters=3)



        # 为 batch=2 绘制空间分布图
        print("Batch 2 的空间分布图")
        plot_spatial(batch2_data,basis='spatial', color='cell_type', title='Batch 2: Cell Type', save_path=os.path.join(umap_dir, 'batch2_cell_type.png'))
        plot_spatial(batch2_data,basis='spatial', color='xgmi_cluster', title='Batch 2: X_gmi Clusters', save_path=os.path.join(umap_dir, 'batch2_xgmi_cluster.png'))

        # 为 batch=3 绘制空间分布图
        print("Batch 3 的空间分布图")
        plot_spatial(batch1_data,basis='spatial', color='cell_type', title='Batch 3: Cell Type', save_path=os.path.join(umap_dir, 'batch1_cell_type.png'))
        plot_spatial(batch1_data,basis='spatial', color='xgmi_cluster', title='Batch 3: X_gmi Clusters', save_path=os.path.join(umap_dir, 'batch1_xgmi_cluster.png'))


        def cal_metric(adata):
            true_labels = adata.obs['cell_type'].values
            predicted_labels = adata.obs['xgmi_cluster'].values    
            
            # 计算每个指标
            homogeneity = homogeneity_score(true_labels, predicted_labels)
            mutual_info = mutual_info_score(true_labels, predicted_labels)
            v_measure = v_measure_score(true_labels, predicted_labels)
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            ari = adjusted_rand_score(true_labels, predicted_labels)
            return homogeneity,mutual_info,v_measure,ami,nmi,ari
        
        homogeneity1,mutual_info1,v_measure1,ami1,nmi1,ari1 = cal_metric(batch1_data)
        homogeneity2,mutual_info2,v_measure2,ami2,nmi2,ari2 = cal_metric(batch2_data)


        print(f"Batch1_Homogeneity: {homogeneity1}")
        print(f"Batch1_Mutual Information: {mutual_info1}")
        print(f"Batch1_V-Measure: {v_measure1}")
        print(f"Batch1_Adjusted Mutual Information (AMI): {ami1}")
        print(f"Batch1_Normalized Mutual Information (NMI): {nmi1}")
        print(f"Batch1_Adjusted Rand Index (ARI): {ari1}")
        print(f"Batch2_Homogeneity: {homogeneity2}")
        print(f"Batch2_Mutual Information: {mutual_info2}")
        print(f"Batch2_V-Measure: {v_measure2}")
        print(f"Batch2_Adjusted Mutual Information (AMI): {ami2}")
        print(f"Batch2_Normalized Mutual Information (NMI): {nmi2}")
        print(f"Batch2_Adjusted Rand Index (ARI): {ari2}")

        # 创建结果字典
        metrics_data = {
            'Metric': [
                'Homogeneity',
                'Mutual Information',
                'V-Measure',
                'Adjusted Mutual Information (AMI)',
                'Normalized Mutual Information (NMI)',
                'Adjusted Rand Index (ARI)'
            ],
            'Batch1': [
                homogeneity1,
                mutual_info1,
                v_measure1,
                ami1,
                nmi1,
                ari1
            ],
            'Batch2': [
                homogeneity2,
                mutual_info2,
                v_measure2,
                ami2,
                nmi2,
                ari2
            ]
        }

        # 转换为 DataFrame
        df = pd.DataFrame(metrics_data)

        # 保存到 CSV
        output_path = os.path.join(umap_dir, "clustering_metrics.csv")
        df.to_csv(output_path, index=False)
        print('metrics saved in clustering_metrics.csv')