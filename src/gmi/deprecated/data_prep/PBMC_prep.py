import os
import mudata as mu
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# 定义全局变量保存路径
MUDATA_FILE = "/data/share_data/yuytest/gmi_data/pbmc.h5mu"
SAVE_PATH_DF = "/home/yuyipei/graph_mosaic_integration/data/All_nodes.csv"
SAVE_PATH_RESULT_DF = "/home/yuyipei/graph_mosaic_integration/data/Weights.csv"

def log_transform_and_normalize(matrix):
    """对表达矩阵进行 log(X+1) 变换和 max-min 归一化"""
    if isinstance(matrix, csr_matrix):
        matrix = matrix.toarray()  # 转换为稠密矩阵以便进行操作
    matrix = np.log1p(matrix)  # log(X + 1) 变换
    # max-min 归一化
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    matrix = (matrix - min_val) / (max_val - min_val) #x' = (x - min(x)) / (max(x) - min(x))
    return csr_matrix(matrix)

def assign_label(idx):
    """根据索引分配标签"""
    if "batch" in idx:
        if "batch1" in idx:
            return "batch1"
        elif "batch2" in idx:
            return "batch2"
        elif "batch3" in idx:
            return "batch3"
        elif "batch4" in idx:
            return "batch4"
        else:
            return "batch"
    elif "atac" in idx:
        return "F1"
    elif "rna" in idx:
        return "F2"
    elif "protein" in idx:
        return "F3"
    else:
        return "unknown"

def process_mdata(mdata_path):

    mdata = mu.read(mdata_path)

    # 创建df
    combined_index = mdata.obs.index.tolist() + mdata.var.index.tolist()
    df = pd.DataFrame(index=combined_index, columns=["type"])
    df["type"] = df.index.map(assign_label)
    df["idx"] = range(len(df))

    # 保存df
    directory = os.path.dirname(SAVE_PATH_DF)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建路径：{directory}")

    df.to_csv(SAVE_PATH_DF, index=True)
    print(f"`df` 已保存到 {SAVE_PATH_DF}")

    result_df = pd.DataFrame(columns=["batch", "feature", "expression"])

    # 遍历每个模态
    for mod_name in mdata.mod.keys():
        adata_mod = mdata[mod_name]

        # 获取细胞和特征的全局索引
        cell_indices = adata_mod.obs.index
        feature_indices = adata_mod.var.index

        # 获取表达矩阵
        expression_matrix = adata_mod.X
        expression_matrix = log_transform_and_normalize(expression_matrix) #通过是否注释这行信息
        row_indices, col_indices = expression_matrix.nonzero()
        expression_values = expression_matrix.data

        # if isinstance(expression_matrix, csr_matrix):  # 稀疏矩阵
        #     row_indices, col_indices = expression_matrix.nonzero()
        #     expression_values = expression_matrix.data
        # else:  # 稠密矩阵
        #     expression_matrix = csr_matrix(expression_matrix)
        #     row_indices, col_indices = expression_matrix.nonzero()
        #     expression_values = expression_matrix.data

        row_indices = np.asarray(row_indices).flatten()
        col_indices = np.asarray(col_indices).flatten()
        expression_values = np.asarray(expression_values).flatten()

        # 将细胞和特征映射到全局索引表中的序列号
        mapped_batches = df.loc[cell_indices[row_indices], "idx"].values
        mapped_features = df.loc[feature_indices[col_indices], "idx"].values

        # 创建临时 DataFrame
        temp_df = pd.DataFrame({
            "batch": mapped_batches,
            "feature": mapped_features,
            "expression": expression_values
        })

        # 合并到结果 DataFrame
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 保存result_df

    result_df.to_csv(SAVE_PATH_RESULT_DF, index=True)
    print(f"`result_df` 已保存到 {SAVE_PATH_RESULT_DF}")
    print(df)
    print(result_df)

    return df, result_df

# 调用主函数
all_nodes, weights = process_mdata(MUDATA_FILE)


# NODES_PATH = "/home/yuyipei/graph_mosaic_integration/data/All_nodes.csv"
# WEIGHTS_PATH = "/home/yuyipei/graph_mosaic_integration/data/Weights.csv"
# emb = '/home/yuyipei/PBG/final_mapped_embeddings.csv'
# weights = pd.read_csv(WEIGHTS_PATH,index_col=0)
# nodes = pd.read_csv(NODES_PATH,index_col=0)
# embedding = pd.read_csv(emb,index_col=0)
# print(weights)
# print(nodes)
# print(embedding)