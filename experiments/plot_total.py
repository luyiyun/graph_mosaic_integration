import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

root_dir = '/home/yuyipei/graph_mosaic_integration/result'

def process_benchmark_csv_with_row_labels(file_path):
    df = pd.read_csv(file_path) 
    row_labels = df.iloc[:, 0]
    numeric_df = df.iloc[:, 1::2]  
    stats_df = numeric_df.agg(['mean', 'std'], axis=1)
    stats_df.index=row_labels
    new_file_path = file_path.replace('.csv', '_stats.csv')
    stats_df.to_csv(new_file_path, index=True)
    
    print(f"处理完成：{new_file_path}")





# 遍历所有子文件夹
for gmi_folder in os.listdir(root_dir):
    # 判断文件夹名称是否以 "GMI_" 开头
    if gmi_folder.startswith('GMI_'):
        gmi_folder_path = os.path.join(root_dir, gmi_folder)
        if not os.path.isdir(gmi_folder_path):
            continue

        # 用于存储各个日期文件夹中提取的 Series
        series_list = []
        # 遍历该 GMI_ 文件夹下的所有日期子文件夹
        for date_folder in os.listdir(gmi_folder_path):
            date_folder_path = os.path.join(gmi_folder_path, date_folder)
            if not os.path.isdir(date_folder_path):
                continue

            # 拼接 benchmark_results.csv 文件的路径
            csv_file = os.path.join(date_folder_path, 'benchmark_results.csv')
            if not os.path.exists(csv_file):
                print(f"警告：{csv_file} 不存在！")
                continue

            try:
                # 读取 CSV 文件
                # CSV 示例第一行：",Unintegrated,GMI,Metric Type"
                # 这里假定第一列没有列名，后续列依次为 Unintegrated、GMI、Metric Type
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"读取文件 {csv_file} 时出错：{e}")
                continue

            # 将第一列作为行索引（行标签）
            df.set_index(df.columns[0], inplace=True)

            # 判断是否存在 "GMI" 列
            if "GMI" not in df.columns:
                print(f"文件 {csv_file} 中不存在 'GMI' 列！")
                continue

            # 提取 GMI 列（此时 index 为指标名称）
            s = df["GMI"]
            # 保存 Series 时将其列名标记为日期（可选：可以使用文件夹名称或其它信息）
            s.name = date_folder
            series_list.append(s)

        # 判断是否有数据
        if len(series_list) == 0:
            print(f"文件夹 {gmi_folder_path} 中没有找到有效的 benchmark_results.csv 文件。")
            continue

        # 将所有 Series 合并为一个 DataFrame，列为不同日期文件夹的数据
        combined_df = pd.concat(series_list, axis=1)

        # 对每个指标（行）计算均值和标准差
        stats_df = pd.DataFrame({
            'mean': combined_df.mean(axis=1),
            'std': combined_df.std(axis=1)
        })

        # 保存结果到 GMI_ 文件夹下
        output_csv = os.path.join(gmi_folder_path, 'combined_benchmark_results_stats.csv')
        stats_df.to_csv(output_csv)
        print(f"处理完成：{output_csv}")


















for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'combined_benchmark_results.csv':
            file_path = os.path.join(subdir, file)
            process_benchmark_csv_with_row_labels(file_path)


# 要处理的方法列表（根据文件夹名前缀）
methods = ['glue', 'harmony', 'midas', 'scmomat','GMI']
# 定义每个方法下期望的数据集顺序（可根据实际情况调整）
datasets_order = ['bmmc', 'MOP', 'muto', 'pbmc', 'triple']

# 用于存放所有结果，key为 (method, dataset)，value为指标 Series
results = {}

# 遍历result目录下的文件夹
for entry in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, entry)
    if os.path.isdir(folder_path):
        # 检查文件夹名称是否符合 <method>_<dataset> 格式
        m = re.match(r'([^_]+)_([^_]+)', entry)
        if m:
            method, dataset = m.group(1), m.group(2)
            # 如果方法在我们的列表中，继续处理
            if method in methods:
                stats_file = os.path.join(folder_path, 'combined_benchmark_results_stats.csv')
                if os.path.exists(stats_file):
                    # 读取CSV文件
                    df = pd.read_csv(stats_file)
                    # 假设第一列为指标名称（可能列名为 "Unnamed: 0" 或其它），取第一列所有数据作为指标
                    metric_names = df.iloc[:, 0]
                    # 取mean和std两列
                    means = df['mean']
                    stds = df['std']
                    # 格式化成 "mean (std)" 的字符串，保留6位小数
                    formatted = means.map(lambda x: f"{x:.6f}") + " (" + stds.map(lambda x: f"{x:.6f}") + ")"
                    # 将结果保存，index为指标名称
                    results[(method, dataset)] = pd.Series(formatted.values, index=metric_names)
                else:
                    print(f"文件不存在：{stats_file}")

# 如果没有数据则退出
if not results:
    print("没有找到符合条件的结果文件！")
    exit()

# 取所有指标名称（假设各文件指标一致，这里以第一个文件的指标为准）
all_metrics = list(next(iter(results.values())).index)

# 创建一个空DataFrame，行索引为指标名称
combined_df = pd.DataFrame(index=all_metrics)

# 构造多级列，按照指定方法顺序，每个方法下按 datasets_order 排序
cols = []
for method in methods:
    for dataset in datasets_order:
        # 如果该组合存在，则加入，否则留空
        cols.append((method, dataset))
        
# 对每个列填入数据（如果某个方法-数据集组合不存在，则填充空字符串）
for col in cols:
    method, dataset = col
    if (method, dataset) in results:
        combined_df[col] = results[(method, dataset)]
    else:
        combined_df[col] = ""

# 设置多级列索引
combined_df.columns = pd.MultiIndex.from_tuples(combined_df.columns, names=['Method', 'Dataset'])

# 为了写入CSV时包含两行表头，我们手动构造文件
output_file = os.path.join(root_dir, 'combined_all_stats.csv')

with open(output_file, 'w', encoding='utf-8') as f:
    # 第一行：方法名，每个方法重复5次（以逗号分隔，首列为空）
    method_header = ['']  # 第一列为空（指标名称标题）
    for method in methods:
        method_header.extend([method] * len(datasets_order))
    f.write(','.join(method_header) + '\n')
    
    # 第二行：数据集名称，每个方法下5个数据集
    dataset_header = ['']  # 第一列为空
    for method in methods:
        for dataset in datasets_order:
            dataset_header.append(dataset)
    f.write(','.join(dataset_header) + '\n')
    
    # 后面每行写入数据，第一列为指标名称，后面为对应的单元格
    for metric in combined_df.index:
        row = [metric]
        for col in combined_df.columns:
            row.append(str(combined_df.loc[metric, col]))
        f.write(','.join(row) + '\n')

print(f"所有结果已合并，文件保存至：{output_file}")



combined_file = os.path.join(root_dir, 'combined_all_stats.csv')

# 读取合并后的csv文件，注意csv文件前两行为表头，第一列为指标名称
df = pd.read_csv(combined_file, header=[0,1], index_col=0)

# 方法和数据集的顺序
methods = ['glue', 'harmony', 'midas', 'scmomat','GMI']
datasets_order = ['bmmc', 'MOP', 'muto', 'pbmc', 'triple']

# 定义一个函数，用于解析 "mean (std)" 格式的字符串，返回 (mean, std)
def parse_value(s):
    # 使用正则表达式提取数字
    m = re.search(r'([\d\.eE+\-]+)\s*\(([\d\.eE+\-]+)\)', s)
    if m:
        mean_val = float(m.group(1))
        std_val = float(m.group(2))
        return mean_val, std_val
    else:
        return np.nan, np.nan

# 针对每个指标，构造与 DataFrame 结构相同的 mean 和 std 数据
# 我们将创建一个字典，key为指标名称，value为包含 mean 和 std 的DataFrame（多级列：方法和数据集）
metrics = ["Batch correction", "Bio conservation", "Total"]

# 对于每个指标，提取对应的那一行，并解析每个单元格数据
# 为便于绘图，我们构造一个结构：
#   means: DataFrame，index为方法，columns为数据集
#   stds: DataFrame，结构同上
parsed_results = {}

for metric in metrics:
    # 取出该指标对应的那一行，注意行索引为指标名称
    row = df.loc[metric]
    # 初始化存储数据的字典，结构为 {method: {dataset: (mean, std)}}
    data = {method: {} for method in methods}
    
    # 遍历所有列（多级索引），并按照方法和数据集存储解析后的数据
    for (meth, ds), value in row.items():
        # 只处理指定的顺序内的数据
        if (meth in methods) and (ds in datasets_order):
            mean_val, std_val = parse_value(str(value))
            data[meth][ds] = (mean_val, std_val)
            
    # 构造 DataFrame
    means = pd.DataFrame(index=methods, columns=datasets_order, dtype=float)
    stds  = pd.DataFrame(index=methods, columns=datasets_order, dtype=float)
    for meth in methods:
        for ds in datasets_order:
            if ds in data[meth]:
                means.loc[meth, ds] = data[meth][ds][0]
                stds.loc[meth, ds]  = data[meth][ds][1]
            else:
                means.loc[meth, ds] = np.nan
                stds.loc[meth, ds]  = np.nan
    parsed_results[metric] = {'means': means, 'stds': stds}

# 绘图设置
bar_width = 0.15  # 每个柱子的宽度
x = np.arange(len(datasets_order))  # 每个数据集的位置

# 对每个指标生成一张图
for metric in metrics:
    means = parsed_results[metric]['means']
    stds  = parsed_results[metric]['stds']
    
    plt.figure(figsize=(10, 6))
    
    # 对于每个方法，绘制一组柱状图
    for i, method in enumerate(methods):
        # x轴偏移量，确保各方法柱子不重叠
        offset = (i - (len(methods)-1)/2) * bar_width
        plt.bar(x + offset, means.loc[method, datasets_order],
                width=bar_width,
                yerr=stds.loc[method, datasets_order],
                capsize=5,
                label=method)
    
    plt.xticks(x, datasets_order, fontsize=12)
    plt.xlabel("Datasets", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(f"{metric} ", fontsize=16)
    plt.legend(title="Algorithms")
    plt.tight_layout()
    
    # 保存图像至result目录，文件名示例：Batch_correction.png
    # 注意处理文件名中的空格等情况
    metric_filename = metric.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(root_dir, f"{metric_filename}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"{metric} 图像已保存至：{save_path}")