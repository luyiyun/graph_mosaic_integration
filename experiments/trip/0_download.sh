#!/bin/bash


save_dir="/data/share_data/yuytest/gmi_data/unprocessed/trip"
mkdir -p "$save_dir"
cd "$save_dir" || exit 1  

declare -A files=(
    ["10x-ATAC-Brain5k.h5ad"]="http://download.gao-lab.org/GLUE/dataset/10x-ATAC-Brain5k.h5ad"
    ["Luo-2017.h5ad"]="http://download.gao-lab.org/GLUE/dataset/Luo-2017.h5ad"
    ["Saunders-2018.h5ad"]="http://download.gao-lab.org/GLUE/dataset/Saunders-2018.h5ad"
)


download_file() {
    local file=$1
    local url=$2
    local temp_file="${file}.tmp" 


    if [[ -f "$file" ]]; then
        echo "$file exists, passed"
        return 0
    fi

    echo "downloading $file ..."


    curl -L --retry 3 --max-time 60 --fail "$url" -o "$temp_file"
    if [[ $? -ne 0 ]]; then
        echo "donwload $file failed,deleted"
        rm -f "$temp_file" 
        return 1
    fi


    if [[ ! -s "$temp_file" ]]; then
        echo "$file donwload failed,deleted"
        rm -f "$temp_file"
        return 1
    fi


    mv "$temp_file" "$file"
    echo "$file finished"
}


for file in "${!files[@]}"; do
    download_file "$file" "${files[$file]}"
done


cd - || exit 1
