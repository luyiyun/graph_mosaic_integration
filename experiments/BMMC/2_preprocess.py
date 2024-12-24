import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
import mudata as mu
from tqdm import tqdm


rna_file = '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681518_MNC_RNA_counts.tsv'
# adt_file = '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681519_MNC_ADT_counts.tsv'
# hto_file = '/data/share_data/yuytest/gmi_data/unprocessed/GSM3681520_MNC_HTO_counts.tsv'

reader = pd.read_table(rna_file, chunksize=10000)
sarr, index = [], []
for dfi in tqdm(reader):
    sarr_i = sparse.coo_array(dfi.values)
    sarr.append(sarr_i)
    index.append(dfi.index.values)
    # print(dfi.shape)
    # print(dfi.iloc[:5, :5])
    # break
sarr = sparse.vstack(sarr)
index = np.concatenate(index)
column = dfi.column


# with open(rna_file, 'r') as f:
#     rna_header = f.readline().strip().split('\t')  # 读取列名
# rna = pd.read_csv(rna_file, sep='\t', dtype=str, names=rna_header, skiprows=1)  # 跳过表头
# rna_sparse = sparse.csr_matrix(rna.values.astype(float))  # 转换为稀疏矩阵
# rna_adata = ad.AnnData(rna_sparse)
# rna_adata.var_names = rna_header[1:]  # 设置基因名
# rna_adata.obs_names = rna.iloc[:, 0]  # 设置细胞名


# with open(adt_file, 'r') as f:
#     adt_header = f.readline().strip().split('\t')
# adt = pd.read_csv(adt_file, sep='\t', dtype=str, names=adt_header, skiprows=1)
# adt_sparse = sparse.csr_matrix(adt.values.astype(float))
# adt_adata = ad.AnnData(adt_sparse)
# adt_adata.var_names = adt_header[1:]
# adt_adata.obs_names = adt.iloc[:, 0]


# hto = pd.read_csv(hto_file, sep='\t', index_col=0)
# labels = hto.idxmax(axis=0)  # 每列最大值的行名作为标签


# obs = pd.DataFrame({
#     'batch': 1,  # 批次信息，设为1
#     'label': labels
# })

# rna_adata.obs = obs
# adt_adata.obs = obs

# mdata = mu.MuData({"RNA": rna_adata, "ADT": adt_adata})

# output_path = '/data/share_data/yuytest/gmi_data/integrated_mudata_sparse.h5mu'
# mdata.write(output_path)

# print(f"Mudata 文件已保存到: {output_path}")