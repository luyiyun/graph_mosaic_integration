import os
import os.path as osp
import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from mudata import MuData
from scipy import sparse as sp
from tqdm import tqdm

from gmi.mmAAVI.preprocess import log1p_norm, lsi  # 假设这些函数已定义
from gmi.mmAAVI.genomic import Bed
import biothings_client as bc

def create_mosaic_dataset(rna_path: str, adt_path: str) -> MuData:
    # 1. 数据加载（强制唯一变量名）
    print("→ 加载原始数据...")
    adata_rna = sc.read_h5ad(rna_path)
    adata_adt = sc.read_h5ad(adt_path)
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    adata_rna = adata_rna[:, adata_rna.var["highly_variable"]]
    # 预处理变量名（防止MuData合并冲突）
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    adata_rna.var_names = ['rna:' + x for x in adata_rna.var_names]
    adata_adt.var_names = ['adt:' + x for x in adata_adt.var_names]

    # 2. 添加batch信息
    print("→ 添加batch信息...")
    for adata in [adata_rna, adata_adt]:
        adata.obs['batch'] = 1
        adata.obs['batch'] = adata.obs['batch'].astype('category')

    # 3. 创建MuData对象
    print("→ 创建MuData对象...")
    mdata = MuData({'rna': adata_rna, 'adt': adata_adt})
    mdata.var_names_make_unique()
    return mdata


def preprocess_mudata(mdata: MuData) -> MuData:
    """预处理并构建多模态网络"""
    logging.info("Preprocessing dataset...")
    
    # ===== 1. 构建 RNA-ADT 网络 =====
    rna_var = mdata.mod["rna"].var
    rna_symbols = rna_var.index.str.replace("rna:", "", regex=True).tolist()
    # 通过 BioThings 查询基因坐标
    gene_client = bc.get_client("gene")
    gene_info = gene_client.querymany(
        rna_symbols,
        scopes="symbol",
        species="human",
        fields=["genomic_pos"],
        as_dataframe=True,
        df_index=True
    )
    # 处理重复和缺失的符号
    gene_info = gene_info[~gene_info.index.duplicated(keep='first')]  # 去重
    gene_info = gene_info.reindex(rna_symbols)  # 确保与原始顺序一致
    print(gene_info)
    # === 步骤4: 构建坐标数据 ===
    coord_data = []
    for sym in rna_symbols:
        # 提取坐标信息
        row = gene_info.loc[sym]
        chrom = str(row["genomic_pos.chr"]).replace("chr", "") if pd.notna(row["genomic_pos.chr"]) else "."
        start = int(row["genomic_pos.start"]) if pd.notna(row["genomic_pos.start"]) else 0
        end = int(row["genomic_pos.end"]) if pd.notna(row["genomic_pos.end"]) else 0
        
        coord_data.append({
            "chrom": chrom,
            "chromStart": start,
            "chromEnd": end,
            "symbol": sym
        })
    
    # === 步骤5: 创建Bed对象 ===
    rna_bed_df = pd.DataFrame(coord_data).set_index("symbol")
    rna_bed_df["strand"] = gene_info["genomic_pos.strand"].map({1.0: "+", -1.0: "-"})
    rna_bed_df["strand"].fillna("+", inplace=True)
    rna_bed = Bed(rna_bed_df[["chrom", "chromStart", "chromEnd", "strand"]])
    # === 步骤6: 创建adt ===
    adt_var = mdata.mod["adt"].var
    adt_symbols = (
        adt_var["gene_ids"]
        .str.replace("adt:", "", regex=True)
        .str.upper()
        .tolist()
    )
    gene_client = bc.get_client("gene")    
    adt_info = gene_client.querymany(
        adt_symbols,
        scopes="symbol",
        species="human",
        fields=["genomic_pos"],
        as_dataframe=True
    )

    coord_data = []
    for sym in adt_symbols:
        # 提取坐标信息
        row = adt_info.loc[sym]
        chrom = str(row["genomic_pos.chr"]).replace("chr", "") if pd.notna(row["genomic_pos.chr"]) else "."
        start = int(row["genomic_pos.start"]) if pd.notna(row["genomic_pos.start"]) else 0
        end = int(row["genomic_pos.end"]) if pd.notna(row["genomic_pos.end"]) else 0
        
        coord_data.append({
            "chrom": chrom,
            "chromStart": start,
            "chromEnd": end,
            "symbol": sym
        })
    
    # === 步骤5: 创建Bed对象 ===
    adt_bed_df = pd.DataFrame(coord_data).set_index("symbol")
    adt_bed_df["strand"] = adt_info["genomic_pos.strand"].map({1.0: "+", -1.0: "-"})  # 使用正负链信息
    adt_bed_df["strand"].fillna("+", inplace=True)
    adt_bed = Bed(adt_bed_df[["chrom", "chromStart", "chromEnd", "strand"]])

    # === 3. 构建区域关联 ===
    # 定义基因体区域（基因坐标向外扩展 2kb）
    rna_regions = rna_bed.expand(upstream=2000, downstream=0)
    
    # 寻找 ADT 基因在 RNA 基因启动子区域的交集
    overlaps = adt_bed.window_graph(
        rna_regions, 
        window_size=0,
        use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"]
    )
    import ipdb;ipdb.set_trace()
    # 转换为连接矩阵
    rna_adt = overlaps.tocsr()    
    # rna_symbols = var_rna.index.str.upper()  # 假设 RNA 基因名需要大写匹配

    # 构建连接矩阵
    row, col = [], []
    for i, adt_sym in tqdm(enumerate(adt_symbols), total=len(adt_symbols)):
        # 确保 rna_symbols 和 adt_sym 是至少一维的数组
        rna_symbols = np.atleast_1d(rna_symbols)
        adt_sym = np.atleast_1d(adt_sym)

        # 进行基因匹配
        matches = np.where(rna_symbols == adt_sym)[0]
        for j in matches:
            row.append(j)
            col.append(i)
    
    rna_adt = sp.coo_array(
        (np.ones(len(row)), (row, col)),
        shape=(len(rna_symbols), len(adt_symbols))
    )

    # ===== 2. 构建全局网络 =====
    net = sp.block_array([
        [None,        rna_adt],
        [rna_adt.T,   None   ]
    ])

    # ===== 3. 过滤无连接的特征 =====
    mdata.varp["net"] = sp.csr_matrix(net)

    # # ===== 4. 生成嵌入特征 =====
    # n_embeds = 100
    # for mod in ["rna", "adt"]:
    #     adata = mdata.mod[mod]
    #     adata.obsm["log1p_norm"] = np.zeros(adata.shape)
    #     adata.obsm["lsi_pca"] = np.zeros((adata.n_obs, n_embeds))
        
    #     for batch in adata.obs["batch"].unique():
    #         batch_mask = adata.obs["batch"] == batch
    #         X_batch = adata.X[batch_mask]
            
    #         # 对数归一化
    #         adata.obsm["log1p_norm"][batch_mask] = log1p_norm(X_batch)
            
    #         # 降维
    #         if mod == "rna":
    #             adata_cp = ad.AnnData(X_batch)
    #             sc.pp.normalize_total(adata_cp)

    #             sc.pp.log1p(adata_cp)
    #             sc.pp.scale(adata_cp)
    #             sc.tl.pca(adata_cp, n_comps=n_embeds, svd_solver="auto")
    #             embed = adata_cp.obsm["X_pca"]
    #         else:  # ADT 通常用 PCA 或直接使用标准化后数据
    #             embed = log1p_norm(X_batch)  # 如果没有其他方法
                
    #         adata.obsm["lsi_pca"][batch_mask] = embed

    return mdata


def main():
    parser = ArgumentParser()
    RNA_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_RNA_combined.h5ad"
    ADT_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_ADT_combined.h5ad"
    OUTPUT_PATH = "/root/autodl-tmp/Human_Lymph_Node/processed"
    parser.add_argument("--RNA_PATH", default=RNA_PATH)
    parser.add_argument("--ADT_PATH", default=ADT_PATH)
    parser.add_argument("--OUTPUT_PATH", default=OUTPUT_PATH)
    parser.add_argument("--preproc_data_name", default="rna_adt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    os.makedirs(args.OUTPUT_PATH, exist_ok=True)
    
    # 数据加载与预处理
    mdata = create_mosaic_dataset(args.RNA_PATH,args.ADT_PATH)
    mdata = preprocess_mudata(mdata)
    # 保存结果
    output_path = osp.join(args.OUTPUT_PATH, f"{args.preproc_data_name}.h5mu")
    mdata.write(output_path)
    logging.info(f"Saved preprocessed data to {output_path}")
    

if __name__ == "__main__":
    main()