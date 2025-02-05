import os
from argparse import ArgumentParser

import numpy as np
import scanpy as sc
import anndata as ad
from scmidas.config import load_config
from scmidas.model import MIDAS
from scmidas.utils import load_predicted
import lightning as L
import random
import torch


def setup_environment(gpu_id="0", seed=42):
    """
    设置运行环境，并设置全局随机种子。

    参数:
        gpu_id: str, 使用的 GPU ID，默认为 '0'。
        seed: int, 随机种子，默认为 42。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    sc.set_figure_params(figsize=(4, 4))

    # 设置全局随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_midas(data_dir, output_dir, max_epochs=2000):
    """
    训练 MIDAS 模型并生成嵌入表示。

    参数:
        data_dir: str, 数据目录路径。
        output_dir: str, 输出目录路径。
        max_epochs: int, 最大训练轮数，默认为 2000。
    """
    # 加载配置
    configs = load_config()
    # del configs["dims_before_enc_atac"]
    # del configs["dims_after_dec_atac"]


    # 定义各模态的转换规则
    transform = {
        # 'rna': 'log1p',
        "atac": "binarize",
        # 'protein': 'log1p'
    }

    mask_config, data_config = [], []
    batch_names = [
    folder for folder in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, folder))
    ]
    for batch in batch_names:
        path = os.path.join(data_dir, batch)
        mask_i, data_i = {}, {}
        for mod in ["rna", "atac"]:
            mask_i_fn = os.path.join(path, f"{mod}_mask.csv")
            data_i_fn = os.path.join(path, f"{mod}.csv")
            if os.path.exists(mask_i_fn) and os.path.exists(data_i_fn):
                mask_i[mod] = mask_i_fn
                data_i[mod] = data_i_fn
        mask_config.append(mask_i)
        data_config.append(data_i)

    dims_x = {
        "rna": [1677],  # RNA data is represented as a cell x 200 matrix.
        "atac": [
            25734
        ],  # ATAC data is split into multiple chunks with varying dimensions
    }
    # 配置模型
    # Configure MIDAS with the data
    datasets, dims_s, s_joint, combs = MIDAS.configure_data_from_csv(
        data_config, mask_config, transform
    )
    model = MIDAS.configure_data(configs, datasets, dims_x, dims_s, s_joint, combs,batch_size = 8192)

    # 打印模型配置以验证
    print("Model configurations:")
    print(configs)

    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        precision=32,
        strategy='auto',
        num_nodes=1,
        max_epochs=max_epochs,
        log_every_n_steps=5
    )

    # 开始训练
    trainer.fit(model=model)

    # 运行预测，生成嵌入表示
    model.predict(
        output_dir,
        joint_latent=True,
        mod_latent=False,
        impute=False,
        batch_correct=True,
        translate=False,
        input=False,
    )

    # 返回模型对象
    return model


def save_embeddings_to_h5ad(output_dir, h5ad_path, model, seed):
    """
    将 MIDAS 生成的嵌入表示保存为 h5ad 文件。

    参数:
        output_dir: str, MIDAS 输出目录路径。
        h5ad_path: str, 保存的 h5ad 文件路径。
        model: MIDAS 模型对象。
        seed: int, 当前随机种子。
    """
    # 加载联合嵌入
    joint_embeddings = load_predicted(output_dir, model.combs, joint_latent=True,batch_correct=True,translate=True)

    # 如果 h5ad 文件已存在，则加载它
    if os.path.exists(h5ad_path):
        adata = ad.read_h5ad(h5ad_path)
    else:
        # 否则创建一个新的 AnnData 对象
        adata = ad.AnnData(X=np.zeros((joint_embeddings["z"]["joint"].shape[0], 1)))

    # 将当前嵌入保存到 obsm 中，命名为 X_midas_{种子数}
    adata.obsm[f"X_midas_{seed}"] = joint_embeddings["z"]["joint"][:,:model.dim_c]
    adata.obsm[f"X_midas_{seed}_tech"] = joint_embeddings["z"]["joint"][:,model.dim_c:]

    # 保存为 h5ad 文件
    adata.write(h5ad_path)
    print(f"嵌入表示已保存到: {h5ad_path} (种子: {seed})")


def run_midas_pipeline(data_dir, output_dir, h5ad_path, max_epochs=2000):
    """
    运行完整的 MIDAS 流程，包括训练、生成嵌入表示并保存为 h5ad 文件。

    参数:
        data_dir: str, 数据目录路径。
        output_dir: str, 输出目录路径。
        h5ad_path: str, 保存的 h5ad 文件路径。
        max_epochs: int, 最大训练轮数，默认为 2000。
    """
    # 定义 5 个不同的随机种子
    seeds = [1,2,3,4,5]

    for seed in seeds:
        print(f"使用随机种子: {seed}")

        # 设置环境
        setup_environment(seed=seed)

        # 训练模型并生成嵌入表示
        model = train_midas(data_dir, output_dir, max_epochs)

        # 将嵌入表示保存为 h5ad 文件
        save_embeddings_to_h5ad(output_dir, h5ad_path, model, seed)


# 示例调用
if __name__ == "__main__":
    data='MOP'
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"/data/share_data/yuytest/gmi_data/midas/{data}_1",
        help="数据目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/home/yuyipei/graph_mosaic_integration/result/midas_{data}",
        help="输出目录路径",
    )
    parser.add_argument(
        "--h5ad_name",
        type=str,
        default=f"/home/yuyipei/graph_mosaic_integration/result/midas_{data}/embeddings.h5ad",
        help="保存的 h5ad 文件名",
    )
    args = parser.parse_args()
    run_midas_pipeline(args.data_dir, args.output_dir, args.h5ad_name)
