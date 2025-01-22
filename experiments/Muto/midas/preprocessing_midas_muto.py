import os
import numpy as np
import pandas as pd
import mudata as mu
from scipy.sparse import issparse

def prepare_midas_input(mdata, output_dir, modality_names=None, batch_key='batch'):
    """
    将 h5mu 格式的数据转换为 MIDAS 所需的输入格式。

    参数:
        mdata: MuData 对象，包含多模态数据。
        output_dir: str，输出文件的目录路径。
        modality_names: dict，可选，指定每个模态的名称。默认是 {'rna': 'rna', 'atac': 'atac', 'protein': 'protein'}。
        batch_key: str，obs 中表示批次的列名。默认是 'batch'。

    返回:
        data_config: list，数据配置，用于 MIDAS 输入。
        mask_config: list，Mask 配置，用于 MIDAS 输入。
        dims_x: dict，每个模态的维度。
    """
    # 默认模态名称
    if modality_names is None:
        modality_names = {'rna': 'rna', 'atac': 'atac', 'protein': 'protein'}
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据配置和 Mask 配置
    data_config = []
    mask_config = []
    dims_x = {}

    # 遍历每个模态
    for modality, modality_name in modality_names.items():
        if modality not in mdata.mod:
            print(f"Warning: Modality '{modality}' not found in mdata. Skipping.")
            continue

        # 提取数据
        modality_data = mdata.mod[modality].X  # 数据矩阵
        modality_obs = mdata.mod[modality].obs  # 观测信息
        modality_var = mdata.mod[modality].var  # 变量信息

        # 打印数据维度信息
        print(f"Modality {modality_name} data shape: {modality_data.shape}")
        print(f"Modality {modality_name} var index length: {len(modality_var.index)}")

        # 检查数据维度和变量索引是否一致
        if modality_data.shape[1] != len(modality_var.index):
            raise ValueError(
                f"Dimension mismatch in modality {modality_name}: "
                f"data shape {modality_data.shape[1]} != var index length {len(modality_var.index)}"
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

            # 创建 vec 目录
            vec_dir = os.path.join(batch_dir, 'vec')
            os.makedirs(vec_dir, exist_ok=True)

            # 创建模态目录
            modality_dir = os.path.join(vec_dir, modality_name)
            os.makedirs(modality_dir, exist_ok=True)

            # 将每个细胞的数据保存为单独的 CSV 文件
            for i, cell_id in enumerate(batch_indices):
                cell_data = batch_data[i, :].reshape(1, -1)  # 1 x 特征
                np.savetxt(os.path.join(modality_dir, f'{i:04d}.csv'), cell_data, delimiter=',')  # 保存为纯数值 CSV 文件

            # 创建 mask 目录
            mask_dir = os.path.join(batch_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)

            # 创建 Mask 文件（假设所有特征都存在）
            mask_path = os.path.join(mask_dir, f'{modality_name}.csv')
            mask_df = pd.DataFrame(np.ones((1, modality_data.shape[1])), columns=modality_var.index)

            mask_df.to_csv(mask_path, index=True)

            # 更新数据配置和 Mask 配置
            if len(data_config) <= len(batches) - 1:
                data_config.append({modality_name: os.path.join(batch_dir, 'vec', modality_name)})
                mask_config.append({modality_name: mask_path})
            else:
                data_config[len(batches) - 1][modality_name] = os.path.join(batch_dir, 'vec', modality_name)
                mask_config[len(batches) - 1][modality_name] = mask_path

        # 记录模态的维度
        dims_x[modality_name] = [modality_data.shape[1]]

    # 创建 feat 目录
    feat_dir = os.path.join(output_dir, 'feat')
    os.makedirs(feat_dir, exist_ok=True)

    # 创建 feat_dims.toml 文件
    feat_dims_path = os.path.join(feat_dir, 'feat_dims.toml')
    with open(feat_dims_path, 'w') as f:
        for modality, dim in dims_x.items():
            f.write(f'{modality} = {dim}\n')

    # 返回配置
    return data_config, mask_config, dims_x


# 示例调用
if __name__ == "__main__":
    # 设置输入和输出路径
    mdata_path = "/data/share_data/yuytest/gmi_data/muto.h5mu"
    output_dir = "/data/share_data/yuytest/gmi_data/midas/muto"

    # 加载 MuData 对象
    print("Loading MuData object...")
    mdata = mu.read(mdata_path)
    print("MuData loaded successfully!")

    # 调用函数生成 MIDAS 输入
    print("Preparing MIDAS input data...")
    data_config, mask_config, dims_x = prepare_midas_input(
        mdata, output_dir, modality_names={'rna': 'rna', 'atac': 'atac'}
    )

    # 打印生成的配置
    print("Data Config:", data_config)
    print("Mask Config:", mask_config)
    print("Dimensions:", dims_x)