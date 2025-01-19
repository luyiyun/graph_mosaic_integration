import os
import numpy as np
import anndata as ad
import scanpy as sc
import mudata as mu
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker, BioConservation
from gmi import run_benchmark, data_infor_integrate
from gmi.metric import get_X_from_mudata


dataset = "triple"
label = "cell_type"
# 定义结果目录
result_dir = f"/home/yuyipei/graph_mosaic_integration/result/{dataset}_glue"
os.makedirs(result_dir, exist_ok=True)

# 遍历 trip_glue 文件夹中的所有子文件夹，提取 glue_trip_embedding.h5ad 文件
trip_glue_dir = f"/home/yuyipei/graph_mosaic_integration/result/{dataset}_glue"
subfolders = [f for f in os.listdir(trip_glue_dir) if os.path.isdir(os.path.join(trip_glue_dir, f))]

# 初始化一个空的 AnnData 对象来存储合并后的数据
combined_adata = None

for subfolder in subfolders:
    subfolder_path = os.path.join(trip_glue_dir, subfolder)
    h5ad_file = os.path.join(subfolder_path, "glue_trip_embedding.h5ad")
    
    if os.path.exists(h5ad_file):
        print(f"Loading {h5ad_file}...")
        adata = sc.read(h5ad_file)
        
        # 提取子文件夹名称中的数字（如 seed_1 -> 1）
        seed_number = subfolder.split("_")[-1]  # 假设子文件夹名称格式为 "seed_1", "seed_42" 等
        new_key = f"X_glue_{seed_number}"  # 生成唯一的键名
        
        # 重命名 obsm 中的键
        if "X_glue" in adata.obsm:
            adata.obsm[new_key] = adata.obsm.pop("X_glue")
        
        # 如果是第一个文件，初始化 combined_adata
        if combined_adata is None:
            combined_adata = ad.AnnData(obs=adata.obs, obsm=adata.obsm)
        else:
            # 合并 obs 和 obsm
            combined_adata.obs = adata.obs.copy()
            for key in adata.obsm.keys():
                if key in combined_adata.obsm:
                    combined_adata.obsm[key] = adata.obsm[key]
                else:
                    combined_adata.obsm[key] = adata.obsm[key]
    else:
        print(f"File {h5ad_file} not found.")

# 保存合并后的 AnnData 对象
combined_h5ad_path = os.path.join(result_dir, "glue_trip_embedding.h5ad")
combined_adata.write(combined_h5ad_path)
print(f"Combined AnnData saved to {combined_h5ad_path}")

# 加载合并后的 AnnData
adata = sc.read(combined_h5ad_path)

# 加载 MuData
mdata_path = f"/data/share_data/yuytest/gmi_data/{dataset}.h5mu"
mdata = mu.read(mdata_path)
print(mdata)

# 将 AnnData 转换为 MuData
X = get_X_from_mudata(mdata, sparse=True, fillna=0)
mdata = ad.AnnData(X=X)

mdata.obs = adata.obs
mdata.obsm = adata.obsm
mdata.obs['label'] = mdata.obs[label]
print("Converted AnnData to MuData.")

# 定义参数
num_cell = adata.shape[0]  # 使用的细胞数量

# 运行基准测试并保存结果
results_list = []
for key in ['X_glue_1', 'X_glue_2', 'X_glue_3', 'X_glue_4', 'X_glue_5']:
    if key in mdata.obsm:
        print(f"Running benchmark for {key}...")
        result_dirs = os.path.join(result_dir, key)
        os.makedirs(result_dirs, exist_ok=True)

        # 运行基准测试
        bm = Benchmarker(
            mdata,
            batch_key="batch",
            label_key="label",
            embedding_obsm_keys=[key],
            n_jobs=-1,
            bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
        )
        bm.benchmark()

        # 打印详细的结果数据框
        df = bm.get_results(min_max_scale=False)
        df_transposed = df.transpose()
        print(df_transposed)
        df_transposed.to_csv(f"{result_dirs}/benchmark_results.csv")
        print('benchmark result saved')

        # 使用嵌入计算邻接矩阵
        print("[INFO] Calculating neighbors using embeddings...")
        sc.pp.neighbors(mdata, use_rep=key)  # 使用 embeddings 计算邻接矩阵

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
            color=[label],
            ax=axes[1],
            title="UMAP colored by Cell Type",
            show=False,
        )
        plt.tight_layout()
        plt_path = os.path.join(result_dirs, f"umap_plot.png")
        plt.savefig(plt_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[INFO] UMAP plot saved to '{plt_path}'")

        # 加载结果
        result_path = os.path.join(result_dir, key, "benchmark_results.csv")
        if os.path.exists(result_path):
            results = pd.read_csv(result_path, index_col=0)
            results_list.append(results)
            print(f"Loaded results for {key}")
        else:
            print(f"Results not found for {key} at {result_path}")
    else:
        print(f"Skipping {key} as it is not in adata.obsm")

# 合并所有结果
if results_list:
    combined_results = pd.concat(results_list, axis=1)
    combined_results.columns = [f"{key}_{col}" for key, df in zip(
        ['X_glue_1', 'X_glue_2', 'X_glue_3', 'X_glue_4', 'X_glue_5'],
        results_list
    ) for col in df.columns]
    
    # 保存合并后的结果
    combined_results.to_csv(os.path.join(result_dir, "combined_benchmark_results.csv"))
    print("Combined benchmark results saved.")
    
    # 打印结果
    print("Combined Benchmark Results:")
    print(combined_results)
else:
    print("No results to combine.")