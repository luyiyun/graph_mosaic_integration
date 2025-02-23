import os
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from mudata import MuData

def create_gmi_dataset(rna_path: str, adt_path: str) -> MuData:
    """
    修复版本：解决网络构建错误和变量名冲突问题
    """
    # 1. 数据加载（强制唯一变量名）
    print("→ 加载原始数据...")
    adata_rna = sc.read_h5ad(rna_path)
    adata_adt = sc.read_h5ad(adt_path)
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]]
    # 预处理变量名（防止MuData合并冲突）
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    adata_rna.var_names = ['rna:' + x for x in adata_rna.var_names]
    adata_adt.var_names = ['adt:' + x for x in adata_adt.var_names]

    # 2. 添加batch信息
    print("→ 添加batch信息...")
    for adata in [adata_rna, adata_adt]:
        adata.obs['batch'] = 1
        adata.obs['batch'] = adata.obs['batch'].astype('category')

    # 3. 创建MuData对象
    print("→ 创建MuData对象...")
    mdata = MuData({'rna': adata_rna, 'adt': adata_adt})
    
    # 4. 构建关联网络（修复参数错误）
    print("→ 构建特征关联网络...")
    rna_features = [x.replace('rna:', '') for x in mdata.mod['rna'].var_names]
    adt_markers = [x.split('_')[0].replace('adt:', '') for x in mdata.mod['adt'].var_names]

    # 创建关联矩阵（正确参数格式）
    rows, cols = [], []
    for i, adt_marker in enumerate(adt_markers):
        for j, rna_feature in enumerate(rna_features):
            if rna_feature.upper() == adt_marker.upper():
                rows.append(j)
                cols.append(i)
    sub_net = sp.coo_matrix(
        (np.ones(len(rows)), (np.array(rows), np.array(cols))),
        shape=(len(rna_features), len(adt_markers))  # 显式指定形状
    )
    
    # 构建全局网络 [Total_dim x Total_dim]
    total_dim = mdata.n_vars
    global_net = sp.lil_matrix((total_dim, total_dim))  # 使用LIL格式便于填充
    
    # 计算偏移量
    rna_start = 0
    adt_start = len(rna_features)
    
    # 填充RNA-ADT关联
    global_net[rna_start:rna_start+len(rna_features), 
              adt_start:adt_start+len(adt_markers)] = sub_net.tocsr()
    
    # 转换为CSR格式并存储

    mdata.varp['net'] = global_net.tocsr()

    return mdata

def save_mudata(mdata: MuData, output_path: str):
    """保存数据（添加路径检查）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mdata.write(output_path)
    print(f"✓ 数据已保存至：{output_path}")

if __name__ == "__main__":
    # 配置路径
    RNA_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_RNA.h5ad"
    ADT_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_ADT.h5ad"
    OUTPUT_PATH = "/root/autodl-tmp/Human_Lymph_Node/processed/gmi_data.h5mu"
    
    try:
        # 执行处理流程
        mudata = create_gmi_dataset(RNA_PATH, ADT_PATH)
        save_mudata(mudata, OUTPUT_PATH)
        
        # 验证输出
        print("\n验证信息：")
        print(f"总特征数：{mudata.n_vars}")
        print(f"网络矩阵形状：{mudata.varp['net'].shape}")
        print(f"非零元素数量：{mudata.varp['net'].nnz}")
        
    except Exception as e:
        print(f"× 处理失败：{str(e)}")