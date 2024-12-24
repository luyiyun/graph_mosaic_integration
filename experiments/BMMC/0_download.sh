#!/bin/bash

# 设置目标目录
TARGET_DIR="/data/share_data/yuytest/gmi_data/unprocessed"
mkdir -p $TARGET_DIR

# 定义数据集和下载
declare -A datasets
datasets=(
    ["GSE156477"]="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156477&format=file"
    ["GSE128639"]="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE128639&format=file"
)
# 循环处理每个数据集
for dataset in "${!datasets[@]}"; do
    URL=${datasets[$dataset]}
    OUTPUT_FILE="$TARGET_DIR/${dataset}_RAW.tar"
    echo "Downloading $dataset dataset..."
    # 下载
    wget --no-check-certificate -O $OUTPUT_FILE $URL
    # 检查下载是否成功
    if [ $? -ne 0 ]; then
        echo "Download of $dataset failed!"
        exit 1
    fi
    # 解压下载的tar文件
    echo "Extracting $dataset dataset..."
    tar -xvf $OUTPUT_FILE -C $TARGET_DIR
    # 检查解压是否成功
    if [ $? -ne 0 ]; then
        echo "Extraction of $dataset failed!"
        exit 1
    fi
    # 删除原始压缩文件（可选）
    rm $OUTPUT_FILE
    echo "$dataset download and extraction completed successfully!"
done

echo "All datasets have been processed!"
echo "Extracting .gz files..."
cd $TARGET_DIR

for file in *.gz; do
    gunzip "$file"
    if [ $? -ne 0 ]; then
        echo "Failed to extract $file!"
        exit 1
    fi
done

echo "All datasets and compressed files have been processed successfully!"