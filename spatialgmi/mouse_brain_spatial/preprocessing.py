import os
import scanpy as sc
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from pybedtools import BedTool

# 读取数据
rna = sc.read("/root/autodl-tmp/new_data/Chen-RNA.h5ad")
atac = sc.read("/root/autodl-tmp/new_data/Chen-2019-ATAC.h5ad")
spatial_zt4 = sc.read("/root/autodl-tmp/new_data/GSM6704280_ZT4_F_Saline_filtered_feature_bc_matrix_processed.h5ad")
spatial_zt14 = sc.read("/root/autodl-tmp/new_data/GSM6704282_ZT14_F_Saline_filtered_feature_bc_matrix_processed.h5ad")
def check_celltypes(adata, obs_key):
    try:
        types = adata.obs[obs_key]
        print(f"发现 {len(types.cat.categories)} 个类别:")
        print(types.value_counts())
        print("\n具体类别名称:")
        print(list(types.cat.categories))
        print("="*50)
    except KeyError:
        print(f"警告: 未找到 {obs_key} 列")
    except AttributeError:
        print(f"注意: {obs_key} 列不是分类类型，正在转换...")
        adata.obs[obs_key] = adata.obs[obs_key].astype('category')
        check_celltypes(adata, obs_key)

# 检查RNA数据
print("RNA细胞类型分布 (cell_type1):")
check_celltypes(rna, 'cell_type1')
save_dir = "/root/graph_mosaic_integration/spatialgmi/mouse_brain_spatial/result/umap/test_for_plot"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
sc.settings.figdir = save_dir
# 绘制ZT4的空间分布
sc.pl.embedding(
    spatial_zt4,
    basis="spatial",
    color="cell_type",
    title="ZT4 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",  # 如果标签太多可以移除或调整位置
    save="ZT4_1.png"
)

# 绘制ZT14的空间分布
sc.pl.embedding(
    spatial_zt14,
    basis="spatial",
    color="cell_type",
    title="ZT14 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",
    save="ZT14_1.png"
)
# 检查ATAC数据
print("\nATAC细胞类型分布 (cell_type):")
check_celltypes(atac, 'cell_type')

# 检查空间数据ZT4
print("\n空间数据ZT4细胞类型分布:")
check_celltypes(spatial_zt4, 'cell_type')

# 检查空间数据ZT14
print("\n空间数据ZT14细胞类型分布:")
check_celltypes(spatial_zt14, 'cell_type')
# 检查RNA与空间数据的基因匹配

# 定义统一映射规则（根据数据特征调整）
unified_mapping = {
    # RNA数据映射
    'RNA': {
        'Astro': 'Astrocyte',
        'GABA.*': 'Neuron',  # 匹配所有GABA开头的类型
        'Glu.*': 'Neuron',    # 匹配所有Glu开头的类型
        'OPC': 'Oligodendrocyte',
        'POPC': 'Oligodendrocyte',
        'Micro': 'Microglia', 
        'Endo1': 'Endothelial',
        'Endo2': 'Endothelial'
    },
    # ATAC数据映射
    'ATAC': {
        'Ast': 'Astrocyte',
        'E.*': 'Neuron',      # 匹配所有E开头的类型
        'Oli.*': 'Oligodendrocyte', # 匹配Oli开头的类型
        'Mic': 'Microglia',
        'Endo': 'Endothelial'
    },
    # 空间数据映射（保持原名称）
    'SPATIAL': {
        'Astrocyte': 'Astrocyte',
        'Neuron': 'Neuron',
        'Oligodendrocyte': 'Oligodendrocyte',
        'Microglia': 'Microglia',        # 如果存在
        'Endothelial': 'Endothelial'      # 如果存在
    }
}

# 定义最终保留的细胞类型（根据实际映射结果调整）
final_celltypes = ['Astrocyte', 'Neuron', 'Oligodendrocyte']

# 修改后的完整函数
def unify_celltypes(adata, dataset_type):
    """统一细胞类型并过滤"""
    # 确定细胞类型列名
    ct_column = 'cell_type1' if dataset_type == 'RNA' else 'cell_type'
    
    # 获取细胞类型数据
    cell_types = adata.obs[ct_column].astype(str)
    
    # 初始化统一类型列
    adata.obs['unified_type'] = None
    
    # 应用映射规则
    mapping = unified_mapping[dataset_type]
    for pattern, unified in mapping.items():
        if dataset_type in ['RNA', 'ATAC']:
            # 正则表达式匹配
            mask = cell_types.str.contains(f'^{pattern}$', regex=True)
        else:
            # 精确匹配
            mask = (cell_types == pattern)
        
        adata.obs.loc[mask, 'unified_type'] = unified
    
    # 过滤未映射的细胞
    before = adata.shape[0]
    adata = adata[~adata.obs['unified_type'].isna()].copy()
    after = adata.shape[0]
    print(f"{dataset_type}过滤: {before} → {after} cells ({after/before:.1%})")
    
    # 保留目标类型
    adata = adata[adata.obs['unified_type'].isin(final_celltypes)].copy()
    adata.obs['cell_type'] = adata.obs['unified_type'].astype('category')
    return adata

# 应用统一规则
print("\n开始统一细胞类型...")
# RNA数据处理
rna = unify_celltypes(rna, 'RNA')
# ATAC数据处理 
atac = unify_celltypes(atac, 'ATAC')
# 空间数据处理
spatial_zt4 = unify_celltypes(spatial_zt4, 'SPATIAL')
spatial_zt14 = unify_celltypes(spatial_zt14, 'SPATIAL')

sc.pl.embedding(
    spatial_zt4,
    basis="spatial",
    color="cell_type",
    title="ZT4 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",  # 如果标签太多可以移除或调整位置
    save="ZT4_2.png"
)

# 绘制ZT14的空间分布
sc.pl.embedding(
    spatial_zt14,
    basis="spatial",
    color="cell_type",
    title="ZT14 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",
    save="ZT14_2.png"
)


# 验证最终类型分布
print("\n统一后的细胞类型分布:")
print("RNA:", rna.obs['cell_type'].value_counts())
print("ATAC:", atac.obs['cell_type'].value_counts())
print("ZT4:", spatial_zt4.obs['cell_type'].value_counts())
print("ZT14:", spatial_zt14.obs['cell_type'].value_counts())

# 更新MuData（需重新创建）
mdata = md.MuData({
    "rna": rna,
    "atac": atac,
    "spatial_zt4": spatial_zt4,
    "spatial_zt14": spatial_zt14
})

print(mdata)

def get_common_genes(adata1, adata2):
    return set(adata1.var_names) & set(adata2.var_names)

# RNA与两个空间数据的共同基因
rna_spatial_zt4_common = get_common_genes(rna, spatial_zt4)
rna_spatial_zt14_common = get_common_genes(rna, spatial_zt14)
print(f"RNA与ZT4空间共同基因数: {len(rna_spatial_zt4_common)}")
print(f"RNA与ZT14空间共同基因数: {len(rna_spatial_zt14_common)}")

# 两个空间数据之间的共同基因
spatial_common = get_common_genes(spatial_zt4, spatial_zt14)
print(f"两个空间数据之间的共同基因数: {len(spatial_common)}")

# 提取最终共同基因（取三者交集）
final_common_genes = rna_spatial_zt4_common & rna_spatial_zt14_common & spatial_common
print(f"最终保留的共同基因数: {len(final_common_genes)}")

# 过滤数据
rna = rna[:, list(final_common_genes)].copy()
spatial_zt4 = spatial_zt4[:, list(final_common_genes)].copy()
spatial_zt14 = spatial_zt14[:, list(final_common_genes)].copy()

# 修正后的ATAC数据坐标处理部分
# 直接使用现有坐标列（注意列名可能不同）
atac.var['chrom'] = atac.var['chrom']  # 如果列名是chromosome需要修改
atac.var['start'] = atac.var['chromStart'].astype(int)
atac.var['end'] = atac.var['chromEnd'].astype(int)


rna.obs_names = "rna_" + rna.obs_names.astype(str)            # RNA 数据添加前缀
spatial_zt4.obs_names = "spatial_zt4_" + spatial_zt4.obs_names.astype(str)  # ZT4 添加前缀
spatial_zt14.obs_names = "spatial_zt14_" + spatial_zt14.obs_names.astype(str)  # ZT14 添加前缀

# 添加 batch 信息
rna.obs['batch'] = '3'  # rna 是 batch3
atac.obs['batch'] = '1'  # rna 是 batch3
spatial_zt4.obs['batch'] = '1'  # spatial_zt4 是 batch1
spatial_zt14.obs['batch'] = '2'  # spatial_zt14 是 batch2

combined_rna = ad.concat(
    [rna, spatial_zt4, spatial_zt14],
    join='inner',  # 只保留共同基因
    label='batch',  # 添加 batch 信息
    keys=['3', '1', '2']  # 对应 rna、spatial_zt4、spatial_zt14
)
combined_rna = combined_rna[~combined_rna.obs_names.duplicated()]

# 处理 obsm['spatial']
# 初始化一个空的数组用于存储空间坐标
spatial_coords = np.empty((combined_rna.n_obs, 2))  # 假设空间坐标是二维的
spatial_coords[:] = np.nan  # 用 NaN 填充

# 将 spatial_zt4 和 spatial_zt14 的空间坐标填充到 combined_rna 中
if 'spatial' in spatial_zt4.obsm:
    spatial_zt4_indices = combined_rna.obs_names.isin(spatial_zt4.obs_names)
    spatial_coords[spatial_zt4_indices] = spatial_zt4.obsm['spatial']

if 'spatial' in spatial_zt14.obsm:
    spatial_zt14_indices = combined_rna.obs_names.isin(spatial_zt14.obs_names)
    spatial_coords[spatial_zt14_indices] = spatial_zt14.obsm['spatial']

# 将空间坐标添加到 combined_rna 的 obsm 中
combined_rna.obsm['spatial'] = spatial_coords
sc.pl.embedding(
    combined_rna[combined_rna.obs['batch'] == '1'],
    basis="spatial",
    color="cell_type",
    title="ZT4 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",  # 如果标签太多可以移除或调整位置
    save="ZT4_3.png"
)

# 绘制ZT14的空间分布
sc.pl.embedding(
    combined_rna[combined_rna.obs['batch'] == '2'],
    basis="spatial",
    color="cell_type",
    title="ZT14 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",
    save="ZT14_3.png"
)
#===========================================================================
#变量筛选
#===========================================================================

sc.pp.highly_variable_genes(combined_rna, flavor="seurat_v3", n_top_genes=3000)
combined_rna = combined_rna[:, combined_rna.var["highly_variable"]]
sc.pp.highly_variable_genes(atac, flavor="seurat_v3", n_top_genes=3000)
atac = atac[:, atac.var["highly_variable"]]
#===========================================================================
#变量筛选
#===========================================================================

mdata = md.MuData({
    'rna': combined_rna,
    'atac': atac
})

print(mdata)

sc.pl.embedding(
    mdata.mod['rna'][mdata.mod['rna'].obs['batch'] == '1'],
    basis="spatial",
    color="cell_type",
    title="ZT4 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",  # 如果标签太多可以移除或调整位置
    save="ZT4_4.png"
)

# 绘制ZT14的空间分布
sc.pl.embedding(
    mdata.mod['rna'][mdata.mod['rna'].obs['batch'] == '2'],
    basis="spatial",
    color="cell_type",
    title="ZT14 Spatial Cell Type Distribution",
    frameon=False,
    legend_loc="on data",
    save="ZT14_4.png"
)


# 创建基因TSS区间（需要获取真实基因坐标）
# 使用mygene获取基因坐标（需要先安装：pip install mygene）
import mygene
mg = mygene.MyGeneInfo()

# 获取基因的TSS信息
gene_symbols = combined_rna.var_names

gene_info = mg.querymany(gene_symbols, scopes='symbol', species='mouse', fields='genomic_pos')

# 构建基因坐标DataFrame
genes_bed = []
for g in gene_info:
    if 'genomic_pos' in g:
        if isinstance(g['genomic_pos'], list):  # 处理多个坐标的情况
            pos = g['genomic_pos'][0]
        else:
            pos = g['genomic_pos']
        genes_bed.append((
            pos['chr'],
            pos['start'] - 2000,  # TSS上游2kb
            pos['start'] + 2000,   # TSS下游2kb
            g['query']
        ))

genes_bed = pd.DataFrame(genes_bed, columns=['chrom', 'start', 'end', 'gene'])
genes_bed = genes_bed.drop_duplicates('gene')

# 创建BedTool对象
atac_bed = BedTool.from_dataframe(atac.var[['chrom', 'start', 'end']])
genes_bedtool = BedTool.from_dataframe(genes_bed)

# 寻找重叠区域
overlaps = atac_bed.intersect(genes_bedtool, wa=True, wb=True)

# 构建关联网络
edges = []
for overlap in overlaps:
    peak_id = overlap.fields[3]  # 假设第4个字段是peak ID
    gene_name = overlap.fields[7]  # 基因名称在第8个字段
    if gene_name in combined_rna.var_names:
        peak_idx = atac.var.index.get_loc(peak_id)
        gene_idx = combined_rna.var_names.get_loc(gene_name)  # 使用 combined_rna 的索引
        edges.append((peak_idx, gene_idx))

# 转换为稀疏矩阵
from scipy.sparse import coo_array, block_array
import numpy as np

# 构建连接矩阵
rows, cols = [], []
for peak_idx, gene_idx in edges:
    rows.append(peak_idx)
    cols.append(gene_idx)

# 创建coo格式的稀疏矩阵
peak_gene_net = coo_array(
    (np.ones(len(edges)), (rows, cols)),
    shape=(atac.n_vars, combined_rna.n_vars)  # 使用 combined_rna 的变量数量
)

# 构建全局网络矩阵
from scipy.sparse import csr_matrix, bmat

# 获取各模态的变量数量
n_vars_rna = combined_rna.n_vars  # 使用 combined_rna 的变量数量
n_vars_atac = atac.n_vars

# 构建全局网络矩阵
net = block_array([
    [None,        peak_gene_net],
    [peak_gene_net.T,   None   ]
])

# 检查维度是否匹配
assert net.shape == (n_vars_rna + n_vars_atac, n_vars_rna + n_vars_atac), "net 矩阵维度不匹配"

# 更新 MuData
mdata.varp["net"] = net.tocsr()

# 保存结果
mdata.write("/root/autodl-tmp/new_data/integrated_data.h5mu")
print("数据处理完成，结果已保存为 integrated_data.h5mu")