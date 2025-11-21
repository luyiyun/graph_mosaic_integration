#!/bin/bash

TARGET_DIR="/data/share_data/yuytest/gmi_data/unprocessed"
mkdir -p $TARGET_DIR

declare -A datasets
datasets=(
    ["GSE156477"]="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE156477&format=file"
    ["GSE128639"]="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE128639&format=file"
)
for dataset in "${!datasets[@]}"; do
    URL=${datasets[$dataset]}
    OUTPUT_FILE="$TARGET_DIR/${dataset}_RAW.tar"
    echo "Downloading $dataset dataset..."
    wget --no-check-certificate -O $OUTPUT_FILE $URL
    if [ $? -ne 0 ]; then
        echo "Download of $dataset failed!"
        exit 1
    fi
    echo "Extracting $dataset dataset..."
    tar -xvf $OUTPUT_FILE -C $TARGET_DIR
    if [ $? -ne 0 ]; then
        echo "Extraction of $dataset failed!"
        exit 1
    fi
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