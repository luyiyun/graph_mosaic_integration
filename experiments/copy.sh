#!/bin/bash

# 定义源目录和目标目录
BASE_DIR="/home/yuyipei/graph_mosaic_integration/result"
TARGET_DIR="$BASE_DIR/total_result"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历 BASE_DIR 下的所有内容，只处理目录
for method_dir in "$BASE_DIR"/*; do
    if [ -d "$method_dir" ]; then
        method_name=$(basename "$method_dir")
        echo "处理方法_数据集文件夹：$method_name"

        # 在 TARGET_DIR 下创建对应的子文件夹
        new_method_dir="$TARGET_DIR/$method_name"
        mkdir -p "$new_method_dir"

        # 遍历该方法_数据集文件夹下的所有子文件夹
        for sub_dir in "$method_dir"/*; do
            if [ -d "$sub_dir" ]; then
                sub_name=$(basename "$sub_dir")
                # 检查该子文件夹中是否存在 benchmark_results.csv 文件
                if [ -f "$sub_dir/benchmark_results.csv" ]; then
                    echo "    找到 $sub_name 中的 benchmark_results.csv 文件"
                    # 在新方法目录下创建对应的子文件夹
                    new_sub_dir="$new_method_dir/$sub_name"
                    mkdir -p "$new_sub_dir"
                    # 将 CSV 文件复制到对应新子文件夹中
                    cp "$sub_dir/benchmark_results.csv" "$new_sub_dir/"
                fi
                
                # 检查该子文件夹中是否存在 umap_plot.png 文件
                if [ -f "$sub_dir/umap_plot.png" ]; then
                    echo "    找到 $sub_name 中的 umap_plot.png 文件"
                    # 将 umap_plot.png 文件复制到对应新子文件夹中
                    cp "$sub_dir/umap_plot.png" "$new_sub_dir/"
                fi
            fi
        done
    fi
done

echo "复制完成！"
