import anndata as ad
import scanpy as sc

RNA_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_RNA.h5ad"
ADT_PATH = "/root/autodl-tmp/Human_Lymph_Node/adata_ADT.h5ad"

adata_rna = sc.read_h5ad(RNA_PATH)
adata_adt = sc.read_h5ad(ADT_PATH)

RNA_PATH2 = "/root/autodl-tmp/Human_Lymph_Node_D1/adata_RNA.h5ad"
ADT_PATH2 = "/root/autodl-tmp/Human_Lymph_Node_D1/adata_ADT.h5ad"

adata_rna2 = sc.read_h5ad(RNA_PATH2)
adata_adt2 = sc.read_h5ad(ADT_PATH2)
import scanpy as sc
import anndata as ad

# 处理前先确保所有变量名唯一
adata_rna.var_names_make_unique()
adata_adt.var_names_make_unique()
adata_rna2.var_names_make_unique()
adata_adt2.var_names_make_unique()

# 添加后缀确保所有细胞索引唯一
def make_obs_unique(adata, suffix):
    adata.obs_names = [f"{x}_{suffix}" for x in adata.obs_names]
    return adata


# 重新执行合并操作（此时索引已唯一）
rna_unique_idx = adata_rna2.obs.index.difference(adata_rna.obs.index)
adt_unique_idx = adata_adt2.obs.index.difference(adata_adt.obs.index)

adata_rna_unique = adata_rna2[rna_unique_idx]
adata_adt_unique = adata_adt2[adt_unique_idx]

# 合并数据，保留所有原始的 `var`
adata_rna_combined = ad.concat([adata_rna, adata_rna_unique], axis=0, join="outer")
adata_adt_combined = ad.concat([adata_adt, adata_adt_unique], axis=0, join="outer")

adata_rna_combined.var=adata_rna.var
adata_adt_combined.var= adata_adt.var
# 最终验证
print("是否有重复细胞索引：")
print(f"RNA: {adata_rna.obs.index.has_duplicates}")
print(f"ADT: {adata_adt.obs.index.has_duplicates}")
adata_rna_combined.write("/root/autodl-tmp/Human_Lymph_Node/adata_RNA_combined.h5ad")
adata_adt_combined.write("/root/autodl-tmp/Human_Lymph_Node/adata_ADT_combined.h5ad")