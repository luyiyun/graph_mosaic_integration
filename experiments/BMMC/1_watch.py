import subprocess

# 文件路径列表
file_paths = [
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681518_MNC_RNA_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681519_MNC_ADT_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681520_MNC_HTO_counts.tsv',
    '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681521_MNC_BULK_RNA.tsv'
]

# 遍历每个文件
for file_path in file_paths:
    print(f"\n处理文件: {file_path}")

    # 使用 shell 命令统计行数
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    num_rows = int(result.stdout.split()[0])  # 获取行数

    # 使用 shell 命令统计列数
    result = subprocess.run(['head', '-n', '1', file_path], stdout=subprocess.PIPE, text=True)
    num_cols = len(result.stdout.strip().split('\t'))  # 获取列数

    # 输出文件 shape
    print(f"表格的 shape: ({num_rows}, {num_cols})")

    # 打印前 5 行 5 列
    print("\n前 5 行 5 列数据：")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # 打印列名
                header = line.strip().split('\t')
                print(header[:5])  # 前 20 列
            else:
                data = line.strip().split('\t')
                print(data[:5])  # 前 20 列

            if i >= 5:  # 打印前 20 行后停止
                break
