#!/bin/bash

TARGET_DIR="/data/share_data/yuytest/gmi_data/unprocessed/muto"
mkdir -p $TARGET_DIR

declare -A datasets
datasets=(
    ["Muto-2021-RNA"]="https://download.gao-lab.org/GLUE/dataset/Muto-2021-RNA.h5ad"
    ["Muto-2021-ATAC"]="https://download.gao-lab.org/GLUE/dataset/Muto-2021-ATAC.h5ad"
)

for dataset in "${!datasets[@]}"; do
    URL=${datasets[$dataset]}
    OUTPUT_FILE="$TARGET_DIR/${dataset}.h5ad"

    echo "Downloading $dataset ..."
    wget --no-check-certificate -O $OUTPUT_FILE $URL

    if [ $? -ne 0 ]; then
        echo "Download of $dataset failed!"
        exit 1
    fi

    echo "$dataset downloaded successfully!"
done

echo "All Muto datasets have been downloaded to: $TARGET_DIR"
