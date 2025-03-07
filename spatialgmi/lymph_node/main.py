import os
import numpy as np
import json
from datetime import datetime
import mudata as mu
from gmi import (
    GraphMosaicIntegration,
    run_benchmark,
    data_infor_integrate,
    plot_umap,
)

# 配置参数
CONFIG = {
    "data_path": "/root/autodl-tmp/Human_Lymph_Node/processed/rna_adt.h5mu",
    "result_root": "./results",
    "alpha_values": [0.02],
    "model_params": {
        "num_neg_per_pos": 4,
        "label_smoothing": 0.1,
        "loss_alpha": 0.2,
        "neg_sampling_mode": "matched",
        "adversarial_balance_weights": False,
        "num_epochs": 100,
        "learning_rate": 0.003,
        "distance_threshold":5,
        "sigma":0.5,
        "use_spatial_distance" : True
    }
}

def run_gmi_pipeline():
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
        # 确保存在以下元数据字段
        mdata = data_infor_integrate(
            mdata,
            feature_key="batch",
            saved_feature_name="batch",
            target_attr="obs",
        )
        mdata.obs['x']=mdata['rna'].obsm['spatial'][:,0]
        mdata.obs['y']=mdata['rna'].obsm['spatial'][:,1]
        # mdata['rna'].obs['x']=mdata['rna'].obsm['spatial'][:,0]
        # mdata['rna'].obs['y']=mdata['rna'].obsm['spatial'][:,1]
        # mdata['adt'].obs['x']=mdata['adt'].obsm['spatial'][:,0]
        # mdata['adt'].obs['y']=mdata['adt'].obsm['spatial'][:,1]
        # mdata = data_infor_integrate(
        #     mdata,
        #     feature_key="cell_type",  # 根据实际注释字段修改
        #     saved_feature_name="label",
        #     target_attr="obs",
        # )
        # mdata = data_infor_integrate(
        #     mdata,
        #     feature_key="lsi_pca",   # 根据实际嵌入字段修改
        #     saved_feature_name="Unintegrated",
        #     target_attr="obsm",
        #     dim_limit=20,
        # )
        
        # Batch编码映射
        batch_mapping = {cat: idx+1 for idx, cat in enumerate(mdata.obs['batch'].cat.categories)}
        mdata.obs['batch'] = mdata.obs['batch'].map(batch_mapping)
    except KeyError as e:
        print(f"× 元数据字段缺失: {str(e)}")
        return

    # 3. 模型训练
    for alpha in CONFIG['alpha_values']:
        # 创建结果目录
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        result_dir = '/root/graph_mosaic_integration/spatialgmi/newone/result'
        # result_dir = os.path.join(CONFIG['result_root'], f"alpha_{alpha}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        print(f"RNA数据的大小: {mdata['rna'].shape}")
        print(f"ADT数据的大小: {mdata['adt'].shape}")
        print(f"net的维度: {mdata.varp['net'].shape}")
        print(f"Batch信息: {mdata.obs['batch'].value_counts()}")
        print(f"RNA数据摘要:\n{mdata['rna']}")
        print(f"ADT数据摘要:\n{mdata['adt']}")

        # try:
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
            
            # # 可视化
            # plot_umap(mdata, color_key="label", save_path=os.path.join(result_dir, "umap.png"))
            
        # except Exception as e:
        #     print(f"× 训练失败 (alpha={alpha}): {str(e)}")
        #     continue

if __name__ == "__main__":
    # 检查GPU可用性
    import torch
    print(f"可用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 执行流程
    run_gmi_pipeline()
    print("整合分析完成")