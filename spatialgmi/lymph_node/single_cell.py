import os
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix

# 定义常量路径
PREPROCESSED_PATH = "/root/autodl-tmp/GSE112903_stroma_annotated.h5ad"
GENE_ANNO_PATH = "/root/autodl-tmp/GSE112903_Table_S1_Mean_Expression.xlsx"
EXPR_PATH = "/root/autodl-tmp/GSE112903_uninf_lcmv_genesVcells.txt.gz"
META_PATH = "/root/autodl-tmp/GSE112903_uninf_lcmv_metadata.xlsx"
RNA_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_RNA.h5ad"
ADT_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_ADT.h5ad"

def preprocess_and_save_adata():
    """预处理数据并保存AnnData对象"""
    # [1] 读取基因注释
    gene_anno_df = pd.read_excel(GENE_ANNO_PATH, index_col=0)
    
    # [2] 读取表达矩阵和元数据
    expr_df = pd.read_csv(EXPR_PATH, sep="\t", index_col=0).T
    meta_df = pd.read_excel(META_PATH, index_col=0)
    
    # [3] 对齐细胞ID
    common_cells = expr_df.index.intersection(meta_df.index)
    expr_sparse = csr_matrix(expr_df.loc[common_cells].values)
    meta_df = meta_df.loc[common_cells]
    
    # [4] 创建AnnData对象
    adata = ad.AnnData(
        X=expr_sparse,
        obs=meta_df,
        var=pd.DataFrame(index=expr_df.columns),
        dtype='float32'
    )
    
    # [5] 整合基因注释
    expr_genes = adata.var.index.tolist()
    common_genes = list(set(expr_genes) & set(gene_anno_df.index))
    adata.var = gene_anno_df.loc[common_genes].reindex(expr_genes)
    adata.var.fillna(0, inplace=True)
    
    # [6] 统一基因名为大写并保存
    adata.var.index = adata.var.index.str.upper()
    adata.write(PREPROCESSED_PATH)
    return adata

# ================== 主流程 ==================
if __name__ == "__main__":
    # 检查预处理文件是否存在
    if os.path.exists(PREPROCESSED_PATH):
        print(f"检测到预处理文件 {PREPROCESSED_PATH}，直接加载...")
        adata = ad.read_h5ad(PREPROCESSED_PATH)
        adata.var.index = adata.var.index.str.upper()
    else:
        print("未找到预处理文件，开始预处理...")
        adata = preprocess_and_save_adata()
        
    # 加载其他数据集
    adata_rna = sc.read_h5ad(RNA_PATH)
    adata_rna.var.index = adata_rna.var.index.str.upper()
    adata_adt = sc.read_h5ad(ADT_PATH)
    
    # 基因匹配分析
    genes_adata = set(adata.var.index)
    genes_rna = set(adata_rna.var.index)
    common_genes = genes_adata & genes_rna
    
    print(f"\n匹配统计:")
    print(f"adata 基因数: {len(genes_adata)}")
    print(f"rna 数据基因数: {len(genes_rna)}")
    print(f"共有基因数: {len(common_genes)} ({len(common_genes)/len(genes_rna):.1%})")
    
    # 保存匹配基因列表
    pd.Series(list(common_genes)).to_csv("/root/autodl-tmp/matched_genes.csv", index=False)


    # 计算未匹配的基因
    unmatched_genes_adata = genes_adata - genes_rna  # adata中存在但rna中没有的
    unmatched_genes_rna = genes_rna - genes_adata    # rna中存在但adata中没有的

    # 打印基本统计
    print(f"\n未匹配基因统计:")
    print(f"adata 中特有基因数: {len(unmatched_genes_adata)}")
    print(f"rna 数据中特有基因数: {len(unmatched_genes_rna)}")

    # 查看示例基因（adata特有）
    if unmatched_genes_adata:
        print("\n示例 - adata特有的前10个基因:")
        for gene in list(unmatched_genes_adata)[:10]:
            # 获取该基因在注释表中的信息（如果存在）
            gene_info = adata.var.loc[gene] if gene in adata.var.index else "N/A"
            print(f"  - {gene}:")
            print(f"    平均表达（Ccl19hi TRC）: {gene_info.get('Ccl19hi TRC.mean', 'N/A')}")
            print(f"    表达细胞比例（Ccl19hi TRC）: {gene_info.get('Ccl19hi TRC.pct_in', 'N/A')}")

    # 查看示例基因（rna特有）
    if unmatched_genes_rna:
        print("\n示例 - rna数据特有的前10个基因:")
        for gene in list(unmatched_genes_rna)[:10]:
            print(f"  - {gene}")

    # ================== 合并数据集 ==================
    # 提取共有基因列表并去重
    common_genes = list(set(common_genes))  # 使用set去重
    common_genes_sorted = sorted(common_genes)  # 排序

    # 检查基因名是否唯一
    if len(common_genes_sorted) != len(set(common_genes_sorted)):
        print("⚠️ 警告：共有基因列表中存在重复基因名，已去重。")

    # 确保 adata 和 adata_rna 的基因名唯一
    if not adata.var.index.is_unique:
        print("adata 中存在重复基因名，正在去重...")
        adata.var_names_make_unique()  # 去重处理

    if not adata_rna.var.index.is_unique:
        print("adata_rna 中存在重复基因名，正在去重...")
        adata_rna.var_names_make_unique()  # 去重处理

    # 子集化数据集 (保留共有基因)
    # 注意：只保留var.index，其他var列丢弃
    try:
        adata_common = adata[:, common_genes_sorted].copy()
        adata_common.var = pd.DataFrame(index=adata_common.var.index)  # 只保留基因名
    except KeyError as e:
        print(f"adata 子集化失败: {str(e)}")
        print("请检查 adata.var.index 中是否存在重复基因名。")
        raise

    try:
        adata_rna_common = adata_rna[:, common_genes_sorted].copy()
        adata_rna_common.var = pd.DataFrame(index=adata_rna_common.var.index)  # 只保留基因名
    except KeyError as e:
        print(f"adata_rna 子集化失败: {str(e)}")
        print("请检查 adata_rna.var.index 中是否存在重复基因名。")
        raise

    # 检查维度
    print(f"\n子集化后维度:")
    print(f"adata_common: {adata_common.shape}")
    print(f"adata_rna_common: {adata_rna_common.shape}")

    # 重命名细胞ID避免冲突 (可选添加前缀)
    # 同步修改 adata_rna 和 adata_adt 的 obs.index
    new_rna_index = "HumanLN_" + adata_rna_common.obs_names
    adata_rna_common.obs_names = new_rna_index
    adata_adt.obs_names = new_rna_index  # 同步更新 adata_adt 的 obs.index

    adata_common.obs_names = "MouseLN_" + adata_common.obs_names

    # 添加批次信息
    adata_common.obs["batch"] = "2"  # adata 是 batch2
    adata_rna_common.obs["batch"] = "1"  # adata_rna 是 batch1

    # 沿细胞轴合并
    combined_adata = ad.concat(
        [adata_rna_common, adata_common],  # 注意顺序：rna在前，stroma在后
        axis=0,
        join="outer",  # 使用outer确保保留所有观察特征
        merge="unique"  # 处理可能重复的var列
    )

    # 检查合并结果
    print("\n合并后数据集信息:")
    print(combined_adata)
    print(f"总细胞数: {combined_adata.n_obs}")
    print(f"总基因数: {combined_adata.n_vars}")

    # 检查批次分布
    print("\n批次分布:")
    print(combined_adata.obs["batch"].value_counts())

    # 保存合并后的数据
    combined_adata.write("/root/autodl-tmp/combined_rna_stroma.h5ad")
    adata_adt.write("/root/autodl-tmp/combined_adt_stroma.h5ad")
    print("\n已保存合并数据集到 /root/autodl-tmp/combined_rna_stroma.h5ad")

# import ipdb;ipdb.set_trace()