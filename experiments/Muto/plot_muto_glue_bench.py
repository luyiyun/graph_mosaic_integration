import os
import anndata as ad
import scanpy as sc
import mudata as mu
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker, BioConservation
from gmi import run_benchmark, data_infor_integrate
from gmi.metric import get_X_from_mudata
dataset = "muto"
label = "cell_type"
# 加载 AnnData
adata = sc.read(f'/home/yuyipei/graph_mosaic_integration/result/glue_{dataset}/glue_{dataset}_embedding.h5ad')
mdata_path = f"/data/share_data/yuytest/gmi_data/{dataset}.h5mu"
mdata = mu.read(mdata_path)
print(mdata)
# 将 AnnData 转换为 MuData
# 假设 adata 包含多个模态的数据（例如 RNA 和 ATAC），需要手动拆分
# 这里假设 adata 只包含一个模态（例如 RNA），其他模态需要根据实际情况补充
X = get_X_from_mudata(mdata, sparse=True, fillna=0)
mdata = ad.AnnData(X=X)

mdata.obs=adata.obs
mdata.obsm=adata.obsm
mdata.obs['label']=mdata.obs[label]
print("Converted AnnData to MuData.")


# 定义参数
num_cell = adata.shape[0]  # 使用的细胞数量
result_dir = f"/home/yuyipei/graph_mosaic_integration/result/{dataset}_glue"  # 结果保存目录
os.makedirs(result_dir, exist_ok=True)  # 创建结果目录

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
            embedding_obsm_keys=[
                # "Unintegrated",
                # "GMI_full",
                # "GMI_bipartitle",
                # "GMI_matched",
                #"GMI_matched_add_feat_1",
                key,
                #"Harmony",
            ],
            #pre_integrated_embedding_obsm_key="lsi_pca",
            n_jobs=-1,
            bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False,nmi_ari_cluster_labels_leiden=True,)
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