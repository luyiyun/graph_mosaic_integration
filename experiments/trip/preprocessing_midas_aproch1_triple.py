import os
import numpy as np
import pandas as pd
import mudata as mu
from scipy.sparse import issparse

def prepare_midas_input(mdata, output_dir, modality_names=None, batch_key='batch'):
    """
    将 h5mu 格式的数据转换为 MIDAS 所需的输入格式（Option 1: 每个模态和批次保存为一个 CSV 文件）。

    参数:
        mdata: MuData 对象，包含多模态数据。
        output_dir: str，输出文件的目录路径。
        modality_names: dict，可选，指定每个模态的名称。默认是 {'rna': 'rna', 'atac': 'atac', 'protein': 'protein'}。
        batch_key: str，obs 中表示批次的列名。默认是 'batch'。

    返回:
        data_config: list，数据配置，用于 MIDAS 输入。
        dims_x: dict，每个模态的维度。
    """
    # 默认模态名称
    if modality_names is None:
        modality_names = {'rna': 'rna', 'atac': 'atac', 'protein': 'protein'}
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据配置
    data_config = []
    dims_x = {}

    # 遍历每个模态
    for modality, modality_name in modality_names.items():
        if modality not in mdata.mod:
            print(f"警告: 模态 '{modality}' 在 mdata 中未找到，跳过处理。")
            continue

        # 提取数据
        modality_data = mdata.mod[modality].X  # 数据矩阵
        modality_obs = mdata.mod[modality].obs  # 观测信息
        modality_var = mdata.mod[modality].var  # 变量信息

        # 打印数据维度信息
        print(f"模态 {modality_name} 数据形状: {modality_data.shape}")
        print(f"模态 {modality_name} 变量索引长度: {len(modality_var.index)}")

        # 检查数据维度和变量索引是否一致
        if modality_data.shape[1] != len(modality_var.index):
            raise ValueError(
                f"模态 {modality_name} 数据维度不匹配: "
                f"数据形状 {modality_data.shape[1]} != 变量索引长度 {len(modality_var.index)}"
            )

        # 获取批次信息
        batches = modality_obs[batch_key].unique()

        # 遍历每个批次
        for batch in batches:
            batch_indices = modality_obs[modality_obs[batch_key] == batch].index
            # 将字符串索引转换为整数索引
            batch_int_indices = modality_obs.index.get_indexer(batch_indices)
            batch_data = modality_data[batch_int_indices, :]

            # 如果数据是稀疏矩阵，转换为密集矩阵
            if issparse(batch_data):
                batch_data = batch_data.toarray()

            # 创建批次目录
            batch_dir = os.path.join(output_dir, f'batch_{batch}')
            os.makedirs(batch_dir, exist_ok=True)

            # 将批次数据保存为单个 CSV 文件（cell x feature 矩阵）
            batch_csv_path = os.path.join(batch_dir, f'{modality_name}.csv')
            batch_df = pd.DataFrame(batch_data, index=batch_indices, columns=modality_var.index)
            batch_df.to_csv(batch_csv_path, index=True)  # 包含索引和列名

            # 更新数据配置
            if len(data_config) <= len(batches) - 1:
                data_config.append({modality_name: batch_csv_path})
            else:
                data_config[len(batches) - 1][modality_name] = batch_csv_path

        # 记录模态的维度
        dims_x[modality_name] = [modality_data.shape[1]]

    # 返回配置
    return data_config, dims_x


# 示例调用
if __name__ == "__main__":
    # 设置输入和输出路径
    mdata_path = "/data/share_data/yuytest/gmi_data/triple.h5mu"
    output_dir = "/data/share_data/yuytest/gmi_data/midas/triple_1"

    # 加载 MuData 对象
    print("加载 MuData 对象...")
    mdata = mu.read(mdata_path)
    print("MuData 加载成功!")

    # 调用函数生成 MIDAS 输入
    print("准备 MIDAS 输入数据...")
    data_config, dims_x = prepare_midas_input(
        mdata, output_dir, modality_names={'rna': 'rna', 'atac': 'atac', 'met': 'met'}
    )

    # 打印生成的配置
    print("数据配置:", data_config)
    print("维度:", dims_x)