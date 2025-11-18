import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

root_dir = '/root/autodl-tmp/result'

# 处理 combined_benchmark_results.csv，生成 combined_benchmark_results_stats.csv
def process_benchmark_csv_with_row_labels(file_path):
    df = pd.read_csv(file_path) 
    row_labels = df.iloc[:, 0]
    numeric_df = df.iloc[:, 1::2]  
    stats_df = numeric_df.agg(['mean', 'std'], axis=1)
    stats_df.index = row_labels
    new_file_path = file_path.replace('.csv', '_stats.csv')
    stats_df.to_csv(new_file_path, index=True)
    print(f"处理完成：{new_file_path}")

# 处理 GMI 文件夹
for gmi_folder in os.listdir(root_dir):
    if gmi_folder.startswith('GMI_'):
        gmi_folder_path = os.path.join(root_dir, gmi_folder)
        if not os.path.isdir(gmi_folder_path):
            continue

        series_list = []
        for date_folder in os.listdir(gmi_folder_path):
            date_folder_path = os.path.join(gmi_folder_path, date_folder)
            if not os.path.isdir(date_folder_path):
                continue

            csv_file = os.path.join(date_folder_path, 'benchmark_results.csv')
            if not os.path.exists(csv_file):
                continue

            df = pd.read_csv(csv_file)
            df.set_index(df.columns[0], inplace=True)
            if "GMI" not in df.columns:
                continue
            s = df["GMI"]
            s.name = date_folder
            series_list.append(s)

        if len(series_list) == 0:
            continue

        combined_df = pd.concat(series_list, axis=1)
        stats_df = pd.DataFrame({
            'mean': combined_df.mean(axis=1),
            'std': combined_df.std(axis=1)
        })

        output_csv = os.path.join(gmi_folder_path, 'combined_benchmark_results_stats.csv')
        stats_df.to_csv(output_csv)
        print(f"处理完成：{output_csv}")

# 处理其它 method 的 combined_benchmark_results.csv
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'combined_benchmark_results.csv':
            file_path = os.path.join(subdir, file)
            process_benchmark_csv_with_row_labels(file_path)

# 读取所有 combined_benchmark_results_stats.csv
methods = ['glue', 'harmony', 'midas', 'scmomat', 'GAMMI']
datasets_order = ['bmmc', 'MOP', 'muto', 'pbmc', 'triple']

results = {}
for entry in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, entry)
    if os.path.isdir(folder_path):
        m = re.match(r'([^_]+)_([^_]+)', entry)
        if m:
            method, dataset = m.group(1), m.group(2)
            if method == 'GMI':
                method = 'GAMMI'  # 👈 GMI改成UniVerse
            if method in methods:
                stats_file = os.path.join(folder_path, 'combined_benchmark_results_stats.csv')
                if os.path.exists(stats_file):
                    df = pd.read_csv(stats_file)
                    metric_names = df.iloc[:, 0]
                    means = df['mean']
                    stds = df['std']
                    formatted = means.map(lambda x: f"{x:.6f}") + " (" + stds.map(lambda x: f"{x:.6f}") + ")"
                    results[(method, dataset)] = pd.Series(formatted.values, index=metric_names)

if not results:
    print("没有找到符合条件的结果文件！")
    exit()

all_metrics = list(next(iter(results.values())).index)
combined_df = pd.DataFrame(index=all_metrics)

cols = []
for method in methods:
    for dataset in datasets_order:
        cols.append((method, dataset))

for col in cols:
    method, dataset = col
    if (method, dataset) in results:
        combined_df[col] = results[(method, dataset)]
    else:
        combined_df[col] = ""

combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns, names=['Method', 'Dataset'])

# 保存完整 combined_all_stats.csv
output_file = os.path.join(root_dir, 'combined_all_stats.csv')
with open(output_file, 'w', encoding='utf-8') as f:
    method_header = ['']
    for method in methods:
        method_header.extend([method] * len(datasets_order))
    f.write(','.join(method_header) + '\n')
    
    dataset_header = ['']
    for method in methods:
        for dataset in datasets_order:
            dataset_header.append(dataset)
    f.write(','.join(dataset_header) + '\n')
    
    for metric in combined_df.index:
        row = [metric]
        for col in combined_df.columns:
            row.append(str(combined_df.loc[metric, col]))
        f.write(','.join(row) + '\n')

print(f"所有结果已合并，保存至：{output_file}")

# -------------------- 开始绘图 --------------------

combined_file = os.path.join(root_dir, 'combined_all_stats.csv')
df = pd.read_csv(combined_file, header=[0,1], index_col=0)

def parse_value(s):
    m = re.search(r'([\d\.eE+\-]+)\s*\(([\d\.eE+\-]+)\)', str(s))
    if m:
        mean_val = float(m.group(1))
        std_val = float(m.group(2))
        return mean_val, std_val
    else:
        return np.nan, np.nan

metrics = ["Batch correction", "Bio conservation", "Total"]
parsed_results = {}

for metric in metrics:
    row = df.loc[metric]
    data = {method: {} for method in methods}
    for (meth, ds), value in row.items():
        if (meth in methods) and (ds in datasets_order):
            mean_val, std_val = parse_value(value)
            data[meth][ds] = (mean_val, std_val)
            
    means = pd.DataFrame(index=methods, columns=datasets_order, dtype=float)
    stds  = pd.DataFrame(index=methods, columns=datasets_order, dtype=float)
    for meth in methods:
        for ds in datasets_order:
            if ds in data[meth]:
                means.loc[meth, ds] = data[meth][ds][0]
                stds.loc[meth, ds]  = data[meth][ds][1]
    parsed_results[metric] = {'means': means, 'stds': stds}

# 绘图设置
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.8", "grid.linestyle": "--", "grid.alpha": 0.7})
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2
})
# 使用统一配色
def get_palette(n_clusters):
    base_palette = [
        "#cfcedb", "#866465", "#e1bd7c", "#8182b5", "#94acda",
        "#e9b3cd", "#9c7887", "#c78e66", "#6964a5", "#D4AFB0",
        "#EFE2BC", "#D4D29E", "#E9CB8F", "#b4a8c2", "#e1c6aa",
        "#7e9ab0", "#c9ada1", "#adb89f", "#9b8d7e", "#cdbec4",
        "#b3c6d2", "#a4a2d2", "#b79d6b"
    ]
    return base_palette[:n_clusters] if n_clusters <= len(base_palette) else sns.color_palette("husl", n_clusters)

colors = get_palette(len(methods))
bar_width = 0.15
x = np.arange(len(datasets_order))

# 绘制每个单独指标
for metric in metrics:
    means = parsed_results[metric]['means']
    stds = parsed_results[metric]['stds']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means.loc[method, datasets_order],
            width=bar_width,
            color=colors[i % len(colors)],
            yerr=stds.loc[method, datasets_order],
            capsize=3,
            error_kw=dict(ecolor="black", lw=1, capsize=3, capthick=1),
            edgecolor="black",
            linewidth=0.8,
            label=method
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, rotation=0)
    ax.set_ylabel("Score", labelpad=10)
    ax.set_xlabel("Datasets", labelpad=10)
    ax.set_title(metric, pad=15, fontweight="bold")
    ax.legend(
        title="Methods",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        borderaxespad=0
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    metric_filename = metric.replace(" ", "_")
    save_path = os.path.join(root_dir, f"{metric_filename}_nature_style.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ {metric} 单独柱状图保存至：{save_path}")

# 绘制综合横排组合图
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for idx, metric in enumerate(metrics):
    means = parsed_results[metric]['means']
    stds = parsed_results[metric]['stds']
    ax = axes[idx]
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means.loc[method, datasets_order],
            width=bar_width,
            color=colors[i % len(colors)],
            yerr=stds.loc[method, datasets_order],
            capsize=3,
            error_kw=dict(ecolor="black", lw=1, capsize=3, capthick=1),
            edgecolor="black",
            linewidth=0.8,
            label=method if idx == 0 else None
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_order, rotation=45, ha="right")
    ax.set_title(metric, pad=10, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Score")
    sns.despine(ax=ax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    title="Methods",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 0.95, 1])
save_path = os.path.join(root_dir, "benchmark_all_metrics_combined.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
print(f"✅ 综合横排组合大图保存至：{save_path}")
