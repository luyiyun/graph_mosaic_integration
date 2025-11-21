import pandas as pd
import numpy as np
from scipy.io import mmread
import mudata as mu
from scipy.sparse import vstack,csr_matrix, coo_matrix
import scipy.sparse as sp
import anndata as ad
import os
from bed import Bed
import biothings_client as bc
import scanpy as sc
from preprocess import log1p_norm, lsi


# 文件夹路径
bm1_path = '/data/share_data/yuytest/gmi_data/unprocessed/BM1'
bm2_path = '/data/share_data/yuytest/gmi_data/unprocessed/BM2'
output_path = '/data/share_data/yuytest/gmi_data/bmmc.h5mu'

# 导入BM1中的数据
# ADT
def bm1():
    adt_matrix = pd.read_csv(os.path.join(bm1_path, 'bm1_adt_matrix_data.csv'), index_col=0)
    adt_cells = pd.read_csv(os.path.join(bm1_path, 'bm1_adt_cell_info.csv'), header=None, skiprows=1)[0].values
    adt_features = pd.read_csv(os.path.join(bm1_path, 'bm1_adt_feature_info.csv'),header=None,skiprows=1)[0].values

    # ATAC
    atac_matrix = mmread(os.path.join(bm1_path, 'bm1_atac_data_matrix.mtx')).tocsc()
    atac_cells = pd.read_csv(os.path.join(bm1_path, 'bm1_atac_cell_info.csv'), header=None, skiprows=1)[0].values
    atac_features = pd.read_csv(os.path.join(bm1_path, 'bm1_atac_feature_info.csv'), header=None, skiprows=1)[0].values

    # Labels
    labels = pd.read_csv(os.path.join(bm1_path, 'l1.csv'), index_col=0)

    # 创建Mudata对象
    # ADT
    adt_data = mu.AnnData(X=adt_matrix.values.T, obs=pd.DataFrame(index=adt_cells), var=pd.DataFrame(index=adt_features))

    adt_data.obs['label'] = labels.values.flatten()
    adt_data.obs['batch'] = 1

    # ATAC
    atac_data = mu.AnnData(X=atac_matrix.T, obs=pd.DataFrame(index=atac_cells), var=pd.DataFrame(index=atac_features))

    atac_data.obs['label'] = labels.values.flatten()
    atac_data.obs['batch'] = 1

    
    # ======cell_ranger处理方式=======
    sc.pp.normalize_total(atac_data)
    sc.pp.log1p(atac_data)
    sc.pp.highly_variable_genes(atac_data, flavor="cell_ranger", n_top_genes=8000)
    # ======cell ranger处理方式=======

    atac_data = atac_data[:, atac_data.var["highly_variable"]]
    # 合并为Mudata对象
    mdata = mu.MuData({'adt': adt_data, 'atac': atac_data})

    # 保存为mudata格式文件
    mdata.write('/data/share_data/yuytest/gmi_data/unprocessed/BM1/mdata_BM1.h5mu')
    print(mdata)
    print(mdata['adt'])
    print(mdata['atac'])
    print(mdata['adt'].var)
    print(mdata['atac'].var)


def bm2():
    # 加载 ADT 细胞信息并修改格式
    adt_cells = pd.read_csv(
        os.path.join(bm2_path, 'bm2_adt_cell_info.csv'), 
        header=None, 
        sep=',', 
        usecols=[0],
        skiprows=1
    )[0].values

    # 修改 ADT 细胞名称格式
    adt_cells = [cell.split('_')[1].replace('.', '-')  for cell in adt_cells]

    # 加载 ADT 数据和元信息
    adt_counts = pd.read_csv(os.path.join(bm2_path, 'bm2_adt_matrix_data.csv'), index_col=0)
    adt_meta = pd.read_csv(os.path.join(bm2_path, 'bm2_adt_feature_info.csv'), index_col=0)

    # 标签信息
    labels = pd.read_csv(os.path.join(bm2_path, 'l1.csv'), index_col=0)

    # 加载 RNA 矩阵
    rna_matrix = mmread(os.path.join(bm2_path, 'bm2_rna_matrix_data.mtx')).tocsc()

    # 加载 RNA 细胞信息并修改格式
    rna_cells = pd.read_csv(
        os.path.join(bm2_path, 'bm2_rna_cell_info.csv'), 
        header=None, 
        sep=',', 
        usecols=[0],
        skiprows=1
    )[0].values

    # 修改 RNA 细胞名称格式
    rna_cells = [cell.split('_')[1].replace('.', '-')  for cell in rna_cells]

    # 加载 RNA 特征信息
    rna_meta = pd.read_csv(os.path.join(bm2_path, 'bm2_rna_feature_info.csv'), index_col=0)

    # 创建 ADT AnnData 对象
    adt_data = mu.AnnData(
        X=adt_counts.values.T,  # 转置矩阵，确保维度匹配
        obs=pd.DataFrame(index=adt_cells),  # 设置细胞标签
        var=pd.DataFrame(index=adt_meta.index)  # 设置特征标签
    )

    # 添加标签和批次信息
    adt_data.obs['label'] = labels.values.flatten()
    adt_data.obs['batch'] = 2  # BM2 批次为 2

    # 创建 RNA AnnData 对象
    rna_data = mu.AnnData(
        X=rna_matrix.T,  # 转置矩阵
        obs=pd.DataFrame(index=rna_cells),  # 设置细胞标签
        var=pd.DataFrame(index=rna_meta.index)  # 设置特征标签
    )

    # 添加标签和批次信息
    rna_data.obs['label'] = labels.values.flatten()
    rna_data.obs['batch'] = 2  # BM2 批次为 2
    
    # ======cell_ranger处理方式=======
    sc.pp.normalize_total(rna_data)
    sc.pp.log1p(rna_data)
    sc.pp.highly_variable_genes(rna_data, flavor="cell_ranger", n_top_genes=8000)
    # ======cell ranger处理方式=======

    # # ======seurat处理方式=======
    # sc.pp.normalize_total(rna_data)
    # sc.pp.log1p(rna_data)
    # sc.pp.highly_variable_genes(rna_data, n_top_genes=2000, flavor="seurat")
    #  # ======seurat处理方式=======

    rna_data = rna_data[:, rna_data.var["highly_variable"]]
    # 创建 Mudata 对象
    mdata = mu.MuData({'adt': adt_data, 'rna': rna_data})

    # 保存文件
    mdata.write(os.path.join(bm2_path, 'mdata_BM2.h5mu'))

    # 输出检查
    print(mdata)
    print(mdata['adt'])
    print(mdata['rna'])
    print(mdata['adt'].var)
    print(mdata['rna'].var)


def comb():
    bm1_path = '/data/share_data/yuytest/gmi_data/unprocessed/BM1/mdata_BM1.h5mu'
    bm2_path = '/data/share_data/yuytest/gmi_data/unprocessed/BM2/mdata_BM2.h5mu'
    bm1 = mu.read(bm1_path)
    bm2 = mu.read(bm2_path)
 
    # 处理 ADT 模态合并
    adt1 = bm1.mod['adt']
    adt2 = bm2.mod['adt']

    # 提取交集的变量名
    common_vars = adt1.var.index.intersection(adt2.var.index)
    adt1 = adt1[:, common_vars]
    adt2 = adt2[:, common_vars]

    # 合并 ADT 数据
    combined_adt_X = vstack([adt1.X, adt2.X])
    combined_adt_obs = pd.concat([adt1.obs, adt2.obs])
    combined_adt_var = pd.DataFrame(index=common_vars)
    combined_adt = mu.AnnData(X=combined_adt_X, obs=combined_adt_obs, var=combined_adt_var)
    # 提取 ATAC 和 RNA 模态
    atac = bm1.mod['atac']
    rna = bm2.mod['rna']
    if isinstance(combined_adt.X, coo_matrix):
        combined_adt.X = combined_adt.X.tocsr()

    if isinstance(atac.X, coo_matrix):
        atac.X = atac.X.tocsr()

    if isinstance(rna.X, coo_matrix):
        rna.X = rna.X.tocsr()
    # 检查并去除重复的 cell 名称
    if combined_adt_obs.index.duplicated().sum() > 0:
        # 获取非重复的索引位置
        non_duplicate_indices = ~combined_adt_obs.index.duplicated(keep='first')
        combined_adt = combined_adt[non_duplicate_indices,:]


    if rna.obs.index.duplicated().sum() > 0:
        # 获取非重复的索引位置
        rna_non_duplicate_indices = ~rna.obs.index.duplicated(keep='first') 
        rna = rna[rna_non_duplicate_indices, :]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> create network >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    # 创建atac var dataframe

    atac_var_df = atac.var.index.str.split("-", expand=True).to_frame(index=False)
    assert not atac_var_df.isnull().any().any()
    atac_var_df.columns = ["chrom", "chromStart", "chromEnd"]

    atac_var_df["chrom"] = atac_var_df["chrom"].str.slice(start=3)
    atac_var_df[["chromStart", "chromEnd"]] = atac_var_df[
        ["chromStart", "chromEnd"]
    ].astype(int)
    atac_var_df.index = atac.var.index.astype(np.str_)
    atac.var = atac_var_df

    # 创建rna var dataframe
    # use biothings_client to get the location of genes in chromosome
    cache_url = "./mygene_cache.sqlite"
    gene_client = bc.get_client("gene")
    gene_client.set_caching(cache_url)
    gene_meta = gene_client.querymany(
        rna.var.index.values,
        species="human",
        scopes=["symbol"],
        fields=[
            "_score",
            "name",
            "genomic_pos.chr",
            "genomic_pos.start",
            "genomic_pos.end",
            "genomic_pos.strand",
        ],
        as_dataframe=True,
        df_index=True,
    )
    gene_client.stop_caching()
    gene_meta.rename(
        inplace=True,
        columns={
            "_score": "score",
            "genomic_pos.chr": "chrom",
            "genomic_pos.start": "chromStart",
            "genomic_pos.end": "chromEnd",
            "genomic_pos.strand": "strand",
        },
    )
    # only remain autosomal and sex chromosome genes
    gene_meta = gene_meta[
        gene_meta["chrom"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])
    ]
    mask_dup = gene_meta.index.duplicated(keep="first")
    gene_meta_nodup = gene_meta[~mask_dup]
    rna_var_df = gene_meta_nodup.reindex(index=rna.var.index)
    # postprocess the rna_df
    rna_var_df.fillna({"chrom": "."}, inplace=True)
    rna_var_df["strand"] = rna_var_df.strand.replace({1.0: "+", -1.0: "-"}).fillna("+")
    rna_var_df["chromStart"] = rna_var_df.chromStart.fillna(0.0)
    rna_var_df["chromEnd"] = rna_var_df.chromEnd.fillna(0.0)
    # must remain strand，it will be used when Bed.expand
    rna_var_df = rna_var_df[["chrom", "chromStart", "chromEnd", "strand"]]
    rna_var_df["strand"] = rna_var_df["strand"].astype(np.str_)
    rna_var_df.index = rna_var_df.index.astype(np.str_)
    rna.var = rna_var_df

    # protein
    fn = "/data/share_data/yuytest/gmi_data/proteins_alias.txt"
    prot_names = np.loadtxt(fn, dtype="U")
    prot_ori = np.array([line[0].lower() for line in prot_names], dtype=np.str_)
    prot_alias = np.array(
        [",".join(line).lower() for line in prot_names], dtype=np.str_
    )  # 统一为小写
    prot_df = pd.DataFrame(dict(alias=prot_alias), index=prot_ori)
    combined_adt.var = prot_df.reindex(combined_adt.var.index)

    # 1. atac-rna
    bed_rna = Bed(rna.var)
    bed_atac = Bed(atac.var)
    atac_rna = bed_atac.window_graph(
        bed_rna.expand(upstream=2e3, downstream=0),
        window_size=0,
        use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"],
    )
    # 2. rna-protein
    var_protein = combined_adt.var
    var_rna = rna.var
    # var_rna_index = var_rna.index.str.slice(4)
    row, col = [], []
    for i, p_alias in enumerate(var_protein["alias"]):
        for j, g_symbol in enumerate(var_rna.index):
            if not isinstance(p_alias, str):  # 跳过非字符串
                continue
            if (g_symbol in p_alias) or (g_symbol.lower() in p_alias.lower()):
                row.append(j)
                col.append(i)
    row, col = np.array(row), np.array(col)
    rna_protein = sp.coo_array(
        (np.ones_like(row), (row, col)),
        shape=(var_rna.shape[0], var_protein.shape[0]),
    )
    net = sp.block_array(
        [
            [None, atac_rna, None],
            [atac_rna.T, None, rna_protein],
            [None, rna_protein.T, None],
        ]
    )

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< create network <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 

    # 创建新的 MuData，顺序很重要，必须和net中的顺序一致也就是atac-rna-protein
    combined_mdata = mu.MuData({'atac': atac, 'rna': rna, 'adt': combined_adt})
    net = csr_matrix(net)
    combined_mdata.varp["net"] = net
    # 保存合并结果
    combined_mdata.write(output_path)

    # 打印检查
    print(combined_mdata)



def pca(mdata: mu.MuData) -> mu.MuData:
    n_embeds = 20  # 降维目标维度
    for mod_name, adatai in mdata.mod.items():
        adatai.obsm["log1p_norm"] = np.zeros(adatai.shape)
        adatai.obsm["lsi_pca"] = np.zeros((adatai.n_obs, n_embeds))

        for bi in adatai.obs["batch"].unique():
            maski = (adatai.obs["batch"] == bi).values
            datai = adatai.X[maski, :]

            # 1. Log1p 标准化
            adatai.obsm["log1p_norm"][maski, :] = log1p_norm(datai)

            # 2. PCA 或 LSI 降维
            if mod_name == "atac":
                embedi = lsi(datai, n_components=n_embeds, n_iter=15)
            else:
                if sp.issparse(datai):
                    datai = sp.csr_matrix(datai)
                adata_cp = ad.AnnData(datai).copy()
                sc.pp.normalize_total(adata_cp)
                sc.pp.log1p(adata_cp)
                sc.pp.scale(adata_cp)
                sc.tl.pca(adata_cp, n_comps=20, use_highly_variable=False, svd_solver="auto")
                embedi = adata_cp.obsm["X_pca"]

            adatai.obsm["lsi_pca"][maski, :] = embedi

    return mdata


if __name__ == '__main__':
    bm1()
    bm2()
    comb()
    mdata = mu.read(output_path)
    mdata = pca(mdata)
    mdata.write(output_path)
    