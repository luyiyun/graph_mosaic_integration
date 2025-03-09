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
from sklearn.metrics import adjusted_rand_score
# 配置参数


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
        mdata.obs['x'] = mdata['rna'].obsm['spatial'][:, 0]
        mdata.obs['y'] = mdata['rna'].obsm['spatial'][:, 1]

        # Batch编码映射
        batch_mapping = {cat: idx + 1 for idx, cat in enumerate(mdata.obs['batch'].cat.categories)}
        mdata.obs['batch'] = mdata.obs['batch'].map(batch_mapping)
    except KeyError as e:
        print(f"× 元数据字段缺失: {str(e)}")
        return

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
    gmi_model = GraphMosaicIntegration(
        num_neg_per_pos=CONFIG['model_params']['num_neg_per_pos'],
        label_smoothing=CONFIG['model_params']['label_smoothing'],
        alpha=alpha,
        loss_alpha=CONFIG['model_params']['loss_alpha'],
        adversarial_training=True,
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
    )
    
    # 模型训练
    print(f"\n→ 开始训练 (alpha={alpha})")
    gmi_model.fit(
        mdata, 
        batch_key="batch",
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
    for alpha in [0.1]:
        CONFIG = {
            "data_path": "/root/autodl-tmp/Human_Lymph_Node/processed/rna_adt.h5mu",
            "result_root": "./results",
            "model_params": {
                "num_neg_per_pos": 4,
                "label_smoothing": 0.1,
                "loss_alpha": 0.2,
                "alpha":0.1,
                "neg_sampling_mode": "matched",
                "adversarial_balance_weights": False,
                "num_epochs": 100,
                "learning_rate": alpha,
                "distance_threshold": 20,
                "sigma": 0.2,
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

        # # 可视化
        # sc.tl.umap(adata)
        # fig, axs = plt.subplots(2, 4, figsize=(22, 10))
        # axs = axs.flatten()
        # # UMAP可视化
        # sc.pl.umap(adata, color='gmi_clusters', ax=axs[0], title='GMI Clustering (UMAP)', palette='tab20', show=False)

        # # 空间分布可视化
        # sc.pl.embedding(adata, basis='spatial', color='gmi_clusters', ax=axs[1], title='Spatial Distribution', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata, basis='spatial', color='ground_truth', ax=axs[2], title='ground_truth', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata2, basis='spatial', color='SpatialGlue', ax=axs[3], title='SpatialGlue', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata2, basis='spatial', color='MOFA', ax=axs[4], title='MOFA', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata2, basis='spatial', color='Seurat', ax=axs[5], title='Seurat', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata2, basis='spatial', color='StabMap', ax=axs[6], title='StabMap', s=50, palette='tab20', show=False)
        # sc.pl.embedding(adata2, basis='spatial', color='MultiVI', ax=axs[7], title='MultiVI', s=50, palette='tab20', show=False)
        # plt.tight_layout()
        # umap_dir = "/root/graph_mosaic_integration/spatialgmi/newone/umap"
        # os.makedirs(umap_dir, exist_ok=True)
        # plt.savefig(os.path.join(umap_dir, f"umap_result_{alpha}.png"), dpi=300)
        # print(f"\n可视化结果已保存为: {os.path.join(umap_dir, f'umap_result_{alpha}.png')}")
        # plt.show()
    # ------------------------------------------
    # 指标计算模块
    # ------------------------------------------
    print("\n=== 空间自相关与聚类一致性评估 ===")

    # 1. 计算Moran's I空间自相关分数
    try:
        print("计算Moran's I空间自相关...")
        
        # 添加空间邻接关系
        sq.gr.spatial_neighbors(
            adata,
            coord_type='generic',
            spatial_key='spatial',
            delaunay=True,
            n_rings=2
        )
        
        # 将嵌入向量作为虚拟特征进行计算
        n_components = adata.obsm['X_gmi'].shape[1]
        for i in range(n_components):
            adata.obs[f'emb_{i}'] = adata.obsm['X_gmi'][:, i]
        
        # 计算所有嵌入维度的Moran's I
        moran = sq.gr.moran(
            adata,
            vals=[f'emb_{i}' for i in range(n_components)],
            n_jobs=4
        )
        mean_moran = moran['I'].mean()
        print(f"▪ 平均Moran's I = {mean_moran:.4f}")
    except Exception as e:
        print(f"× Moran's I计算失败: {str(e)}")
        mean_moran = np.nan

    # 2. 计算Jaccard相似度与调整Rand指数
    try:
        print("\n计算聚类一致性指标...")
        
        # 定义指标计算函数
        def compute_cluster_metrics(true_labels, pred_labels):
            """计算多种聚类一致性指标"""
            # 最佳匹配Jaccard
            unique_true = np.unique(true_labels)
            unique_pred = np.unique(pred_labels)
            jaccard_scores = []
            
            for t_cluster in unique_true:
                t_mask = true_labels == t_cluster
                max_j = 0
                for p_cluster in unique_pred:
                    p_mask = pred_labels == p_cluster
                    intersection = np.sum(t_mask & p_mask)
                    union = np.sum(t_mask | p_mask)
                    if union > 0:
                        max_j = max(max_j, intersection/union)
                jaccard_scores.append(max_j)
            
            avg_jaccard = np.mean(jaccard_scores)
            
            # 调整Rand指数
            ari = adjusted_rand_score(true_labels, pred_labels)
            
            return avg_jaccard, ari

        # 与其他方法比较
        comparison_methods = {
            'ground_truth': 'ground_truth',
            'SpatialGlue': 'SpatialGlue',
            'MOFA': 'MOFA',
            'Seurat': 'Seurat',
            'StabMap': 'StabMap',
            'MultiVI': 'MultiVI'
        }
        
        # 将比较方法数据整合到当前adata
        for k in comparison_methods:
            adata.obs[k] = adata2.obs[comparison_methods[k]].astype('category')
        
        # 计算所有对比指标
        metrics_results = {}
        for method in comparison_methods:
            jaccard, ari = compute_cluster_metrics(
                adata.obs['gmi_clusters'].values,
                adata.obs[method].values
            )
            metrics_results[method] = {
                'jaccard': jaccard,
                'ari': ari
            }
            print(f"▪ 与{method}对比: Jaccard={jaccard:.3f}, ARI={ari:.3f}")

    except Exception as e:
        print(f"× 聚类指标计算失败: {str(e)}")
        metrics_results = {}

    # 3. 保存指标结果
    metrics_dir = "/root/graph_mosaic_integration/spatialgmi/newone/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    result_dict = {
        'moran_i': mean_moran,
        'cluster_metrics': metrics_results,
        'alpha': alpha,
        'timestamp': timestamp
    }

    with open(os.path.join(metrics_dir, f"metrics_alpha{alpha}.json"), 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"✓ 指标结果保存至: {metrics_dir}")

    # 4. 在可视化图中添加指标标注
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    axs = axs.flatten()



    # 在第一个子图添加文本标注
    text_content = [
        f"Moran's I: {mean_moran:.3f}",
        "Clustering Consistency:"
    ]
    for method in comparison_methods:
        if method in metrics_results:
            text_content.append(
                f"{method}: J={metrics_results[method]['jaccard']:.2f}"
            )

    axs[0].text(
        0.05, -0.25,  # 调整文本位置
        "\n".join(text_content),
        transform=axs[0].transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox={'facecolor': 'white', 'alpha': 0.8}
    )
    umap_dir = "/root/graph_mosaic_integration/spatialgmi/newone/umap"
    plt.tight_layout()
    plt.savefig(os.path.join(umap_dir, f"umap_result_{alpha}.png"), dpi=300)
    plt.close()  # 防止图像重复显示
    print("整合分析完成")
