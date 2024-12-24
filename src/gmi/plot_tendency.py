import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(result_path):
    """加载所有结果文件"""
    results = []
    for folder in os.listdir(result_path):
        folder_path = os.path.join(result_path, folder)
        if os.path.isdir(folder_path):
            # 读取parameters.json
            param_file = os.path.join(folder_path, 'args.json')
            with open(param_file, 'r') as f:
                params = json.load(f)
            num_neg_per_pos = params.get("num_neg_per_pos", None)

            # 读取benchmark_results.csv
            benchmark_file = os.path.join(folder_path, 'benchmark_results.csv')
            if os.path.exists(benchmark_file):
                df = pd.read_csv(benchmark_file)
                # 将第一列作为指标名称
                df.rename(columns={df.columns[0]: "Metric"}, inplace=True)
                df['num_neg_per_pos'] = num_neg_per_pos
                results.append(df)

    # 合并所有结果
    all_results = pd.concat(results, ignore_index=True)
    return all_results


def plot_results(all_results, save_path='./results'):
    # 计算均值和标准差
    mean_results = all_results.groupby(["num_neg_per_pos", "Metric"]).agg(
        mean_value=("GMI", "mean"),
        std_value=("GMI", "std")
    ).reset_index()

    # 图1：Total指标随num_neg_per_pos变化
    total_results = mean_results[mean_results["Metric"] == "Total"]
    plt.figure(figsize=(8, 6))
    plt.errorbar(total_results["num_neg_per_pos"], total_results["mean_value"], yerr=total_results["std_value"], fmt='-o', capsize=5, label="Total")
    plt.xlabel("num_neg_per_pos")
    plt.ylabel("GMI Total Score")
    plt.title("Total Score vs. num_neg_per_pos")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'total_score_vs_num_neg_per_pos.png'))

    # 图2：多指标变化
    metrics_to_plot = ["KBET", "Leiden ARI", "Leiden NMI", "Total"]
    plt.figure(figsize=(10, 8))
    for metric in metrics_to_plot:
        metric_results = mean_results[mean_results["Metric"] == metric]
        if metric_results.empty:  # 检查是否为空
            print(f"No data for metric: {metric}")
            continue
        plt.errorbar(metric_results["num_neg_per_pos"], metric_results["mean_value"], yerr=metric_results["std_value"], fmt='-o', capsize=5, label=metric)

    plt.xlabel("num_neg_per_pos")
    plt.ylabel("GMI Score")
    plt.title("Metric Comparison vs. num_neg_per_pos")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'metric_comparison_vs_num_neg_per_pos.png'))


if __name__ == "__main__":
    result_path = '/home/yuyipei/graph_mosaic_integration/pbmc_test_result'  # 存储结果的文件夹
    save_path = '/home/yuyipei/graph_mosaic_integration/result'    # 保存图表的文件夹

    # 加载数据
    all_results = load_results(result_path)

    # 绘制图表
    plot_results(all_results, save_path=save_path)
