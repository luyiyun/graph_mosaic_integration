import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def plot_umap(mdata, result_dir, neg_sampling_mode="matched"):
    """
    评估训练后的嵌入，并使用 UMAP 可视化。

    Parameters:
    ----------
    mdata : mu.MuData
        输入的 MuData 对象，包含多模态数据。
    embeddings : pd.DataFrame
        对齐后的嵌入数据，索引与 mdata.obs.index 对齐。
    result_dir : str
        保存结果的目录路径。
    neg_sampling_mode : str, optional
        嵌入类型模式，用于文件命名，默认为 "matched"。
    """


    # 确保保存目录存在
    os.makedirs(result_dir, exist_ok=True)
    embedding_path = os.path.join(result_dir, "final_embeddings_matched_add_feat.csv")
    embeddings = pd.read_csv(embedding_path, index_col=0)
    # 将对齐后的嵌入加入到 mdata
    mdata.obsm["X_embeddings"] = embeddings.loc[mdata.obs.index].to_numpy()

    # 确保 'batch' 和 'cell_type' 信息已整合到 mdata.obs
    if "batch" not in mdata.obs:
        raise ValueError("batch information is missing in mdata.obs.")
    if "label" not in mdata.obs:
        raise ValueError("cell_type (label) information is missing in mdata.obs.")

    # 使用嵌入计算邻接矩阵
    print("[INFO] Calculating neighbors using embeddings...")
    sc.pp.neighbors(mdata, use_rep="X_embeddings")  # 使用 embeddings 计算邻接矩阵

    # 计算 UMAP
    print("[INFO] Calculating UMAP...")
    sc.tl.umap(mdata)  # 生成 UMAP 表示

    # 绘制并保存 UMAP 图
    print("[INFO] Plotting UMAP...")
    plt.figure(figsize=(18, 8))
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sc.pl.umap(
        mdata,
        color=["batch"],
        ax=axes[0],
        title="UMAP of Batch Embeddings",
        show=False,
    )
    sc.pl.umap(
        mdata,
        color=["label"],
        ax=axes[1],
        title="UMAP colored by Cell Type",
        show=False,
    )
    plt.tight_layout()
    plt_path = os.path.join(result_dir, f"umap_plot_{neg_sampling_mode}.png")
    plt.savefig(plt_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] UMAP plot saved to '{plt_path}'")