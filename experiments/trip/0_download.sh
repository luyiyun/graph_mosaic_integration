#!/bin/bash

# 设置保存目录
save_dir="/data/share_data/yuytest/gmi_data/unprocessed/trip"
mkdir -p "$save_dir"
cd "$save_dir" || exit 1  # 如果进入目录失败，则退出

# 文件列表和下载链接
declare -A files=(
    ["10x-ATAC-Brain5k.h5ad"]="http://download.gao-lab.org/GLUE/dataset/10x-ATAC-Brain5k.h5ad"
    ["Luo-2017.h5ad"]="http://download.gao-lab.org/GLUE/dataset/Luo-2017.h5ad"
    ["Saunders-2018.h5ad"]="http://download.gao-lab.org/GLUE/dataset/Saunders-2018.h5ad"
)

# 下载函数
download_file() {
    local file=$1
    local url=$2
    local temp_file="${file}.tmp"  # 临时文件名

    # 检查文件是否已完整存在
    if [[ -f "$file" ]]; then
        echo "$file 已存在，跳过下载。"
        return 0
    fi

    echo "正在下载 $file ..."

    # 使用 curl 下载到临时文件，支持断点续传和设置超时
    curl -L --retry 3 --max-time 60 --fail "$url" -o "$temp_file"
    if [[ $? -ne 0 ]]; then
        echo "下载 $file 失败，删除不完整文件。"
        rm -f "$temp_file"  # 下载失败，删除临时文件
        return 1
    fi

    # 检查临时文件是否有效（文件大小 > 0）
    if [[ ! -s "$temp_file" ]]; then
        echo "$file 下载失败或为空文件，删除临时文件。"
        rm -f "$temp_file"
        return 1
    fi

    # 下载成功，重命名为目标文件
    mv "$temp_file" "$file"
    echo "$file 下载完成。"
}

# 执行下载
for file in "${!files[@]}"; do
    download_file "$file" "${files[$file]}"
done

# 返回原目录
cd - || exit 1
