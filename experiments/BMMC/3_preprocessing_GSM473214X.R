#options(BioC_mirror = "https://mirrors.sjtug.sjtu.edu.cn/bioconductor/")
options("repos" = c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
required_packages <- c(
  "Seurat", "ensembldb", "BiocGenerics", "GenomicRanges", 
  "IRanges", "GenomeInfoDb", "GenomicFeatures", "AnnotationDbi", "Biobase", 
  "AnnotationFilter", "BSgenome", "Biostrings", "XVector", "rtracklayer", 
  "dplyr", "Matrix", "purrr","rhdf5", "argparse"
)





missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(missing_packages)) install.packages(missing_packages)

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk") 
remotes::install_github("r-lib/ymlthis")

# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# options(repos = c(
#   CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/",
#   BioCsoft = "https://bioconductor.org/packages/3.16/bioc",  # 替换为你对应版本
#   BioCann = "https://bioconductor.org/packages/3.16/data/annotation",
#   BioCexp = "https://bioconductor.org/packages/3.16/data/experiment",
#   BioCworkflows = "https://bioconductor.org/packages/3.16/workflows"
# ))
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
library(GenomeInfoDb)
library(BSgenome.Hsapiens.UCSC.hg38)

lapply(required_packages, library, character.only = TRUE)
#===================================package===========================
library(argparse)
library(Seurat)
library(Signac)
library(rhdf5)
library(Matrix)
library(biovizBase)
source("/home/yuyipei/graph_mosaic_integration/experiments/BMMC/utils.R")


#===================================package===========================

#=================================file path============================

frag_path <- file.path("/data/share_data/yuytest/gmi_data/unprocessed", "GSM4732140_Human_BoneMarrow_hg38_fragments.tsv.gz")
adt_path <- file.path("/data/share_data/yuytest/gmi_data/unprocessed", "GSM4732141_Human_BoneMarrow_ADT.tsv.gz")
hto_path <- file.path("/data/share_data/yuytest/gmi_data/unprocessed", "GSM4732142_Human_BoneMarrow_HTO.tsv.gz")


output_dir <- file.path("/data/share_data/yuytest/gmi_data/unprocessed", "output")

mkdir(output_dir, remove_old = T)
#=================================file path============================

# ADT
# load data
adt_counts <- t(read.table(file = adt_path, sep = "\t", header = TRUE, row.names = 1))
adt <- gen_adt(adt_counts)
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
adt
# QC
adt <- subset(adt, subset = nCount_adt > 150 & nCount_adt < 10000)
VlnPlot(adt, c("nCount_adt"), pt.size = 0.001, ncol = 1, log = T) + NoLegend()
adt


# ATAC
atac <- gen_atac(frag_path)
VlnPlot(atac, c("nCount_atac", "nucleosome_signal", "TSS.enrichment"),
        pt.size = 0.001, ncol = 3, log = T) + NoLegend()
atac
# QC
atac <- subset(atac, subset =
                 nCount_atac > 500 & nCount_atac < 3e4 &
                 nucleosome_signal < 3 &
                 TSS.enrichment > 2
)
atac <- subset(atac, features = rownames(atac)[rowSums(atac$atac@counts > 0) > 5])
VlnPlot(atac, c("nCount_atac", "nucleosome_signal", "TSS.enrichment"),
        pt.size = 0.001, ncol = 3, log = F) + NoLegend()
atac


# Get intersected cells satisfying QC metrics of all modalities
cell_ids <- Reduce(intersect, list(colnames(atac), colnames(adt)))
atac <- subset(atac, cells = cell_ids)
adt <- subset(adt, cells = cell_ids)
atac
adt

browser()
atac <- RunTFIDF(atac) %>%
        FindTopFeatures(min.cutoff = "q0")
# rna <- NormalizeData(rna) %>%
#         FindVariableFeatures(nfeatures = 4000) %>%
#         ScaleData()
VariableFeatures(adt) <- rownames(adt)
adt <- NormalizeData(adt, normalization.method = "CLR", margin = 2) %>%
        ScaleData()
# ==================================adt saveing==================================
cell_info_adt <- rownames(adt@meta.data)
write.csv(cell_info_adt, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_adt_cell_info.csv", row.names = FALSE)
feature_info_adt <- rownames(adt@assays$adt)
write.csv(feature_info_adt, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_adt_feature_info.csv", row.names = FALSE)
matrix_data <- as.matrix(adt@assays$adt@layers$counts)
write.csv(matrix_data, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_adt_matrix_data.csv")
# ==================================adt saveing==================================


# ==================================atac saveing==================================

cell_info_atac <- rownames(atac@meta.data)
write.csv(cell_info_atac, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_atac_cell_info.csv", row.names = FALSE)

feature_info_atac <- rownames(atac@assays$atac)
write.csv(feature_info_atac, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_atac_feature_info.csv", row.names = FALSE)

Matrix::writeMM(atac@assays$atac@data, file = "/data/share_data/yuytest/gmi_data/unprocessed/bm1_atac_data_matrix.mtx")

# ==================================atac saveing==================================



# preprocess and save data
#preprocess(output_dir, atac = atac, adt = adt)





















# #===============================Human Bone Marrow============================================================================
# #===============================ADT============================================================================
# # 处理 ADT 数据
# # 加载 ADT 数据
# adt_counts <- t(read.table(file = adt_path, sep = "\t", header = TRUE, row.names = 1))
# 
# # 创建 Seurat 对象
# adt <- CreateSeuratObject(counts = adt_counts)
# # 获取 ADT 计数矩阵（稀疏矩阵）
# adt_counts <- adt@assays$RNA@layers$counts
# 
# # 计算每个细胞的 nCount_adt（每列的总和）
# adt[["nCount_adt"]] <- Matrix::colSums(adt_counts)
# 
# # 检查 nCount_adt 是否已经添加到元数据中
# head(adt@meta.data)
# 
# # 进行 QC 筛选
# adt <- subset(adt, subset = nCount_adt > 150 & nCount_adt < 10000)
# #===============================ADT============================================================================
# 
# 
# 
# 
# #===============================Atac============================================================================
# # 处理 ATAC 数据
# # 创建 ATAC Fragment 对象
# atac <- read.table(frag_path, header = TRUE, sep = "\t")
# 
# 
# # 重命名列以符合片段数据格式
# colnames(atac) <- c("chr", "start", "end", "barcode", "count")
# 
# fragments <- CreateFragmentObject(
#   path = frag_path, 
#   cells = unique(atac$barcode)
# )
# 
# granges <- GRanges(
#   seqnames = atac$chr,
#   ranges = IRanges(start = atac$start, end = atac$end),
#   strand = "*"
# )
# 
# 
# chrom_assay <- CreateChromatinAssay(
#   counts = NULL,         # ATAC数据无需counts矩阵
#   sep = c(":", "-"),   # 默认分隔符
#   genome = 'hg38',       # 修改基因组版本按需调整
#   fragments = fragments
# )
# 
# atac <- CreateSeuratObject(
#   counts = chrom_assay,
#   assay = "ATAC"
# )
# 
# atac <- NucleosomeSignal(atac)         # 核小体信号
# atac <- TSSEnrichment(atac, fast = FALSE) # TSS富集度
# 
# 
# 
# # 进行 QC 筛选
# atac_seurat <- subset(atac_seurat, subset =
#                         nCount_atac > 500 & nCount_atac < 3e4 &
#                         nucleosome_signal < 3 &
#                         TSS.enrichment > 2)
# 
# # 只保留在 ATAC 数据中具有表达的基因
# atac_seurat <- subset(atac_seurat, features = rownames(atac_seurat)[rowSums(atac_seurat$atac@counts > 0) > 5])
# 
# # 
# # # 获取交集细胞（符合所有 QC 指标的细胞）
# # cell_ids <- Reduce(intersect, list(colnames(atac_seurat), colnames(adt)))
# # atac_seurat <- subset(atac_seurat, cells = cell_ids)
# # adt <- subset(adt, cells = cell_ids)
# #===============================Atac============================================================================
# 
# 
# 
# 
# # 
# # # ATAC 数据
# # # 获取 ATAC 的计数矩阵（稀疏矩阵转换为密集矩阵）
# # atac_counts_dense <- as.matrix(atac_seurat$counts)
# # 
# # # 获取 ATAC 的元数据
# # atac_meta <- atac_seurat@meta.data
# # 
# # # 获取 ATAC 的 active.ident
# # atac_active_ident <- atac_seurat@active.ident
# # 
# # # 保存为 CSV 文件
# # write.table(atac_counts_dense, file = file.path(output_dir, "atac_counts.csv"), sep = ",", row.names = TRUE, col.names = NA)
# # write.table(atac_meta, file = file.path(output_dir, "atac_meta.csv"), sep = ",", row.names = TRUE)
# # write.table(as.data.frame(atac_active_ident), file = file.path(output_dir, "atac_active_ident.csv"), sep = ",", row.names = TRUE)
# # 
# # 
# # 
# # 
# # 
# # # ADT 数据
# # # 获取 ADT 的计数矩阵
# # adt_counts_dense <- as.matrix(adt@assays$RNA@counts)
# # 
# # # 获取 ADT 的元数据
# # adt_meta <- adt@meta.data
# # 
# # # 获取 ADT 的 active.ident
# # adt_active_ident <- adt@active.ident
# # 
# # # 保存为 CSV 文件
# # write.table(adt_counts_dense, file = file.path(output_dir, "adt_counts.csv"), sep = ",", row.names = TRUE, col.names = NA)
# # write.table(adt_meta, file = file.path(output_dir, "adt_meta.csv"), sep = ",", row.names = TRUE)
# # write.table(as.data.frame(adt_active_ident), file = file.path(output_dir, "adt_active_ident.csv"), sep = ",", row.names = TRUE)
# # 
# # 
# # 












# #===============================MNC============================================================================
# #=============================RNA=========================================
# # RNA
# # load data
# rna_counts <- read.table(file = rna_path, sep = "\t", header = TRUE)

# rna <- CreateSeuratObject(counts = rna_counts)

# # 添加质控指标
# rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^MT-")

# # 绘制 QC 指标小提琴图
# VlnPlot(rna, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
#         pt.size = 0.001, ncol = 3, log = FALSE) + NoLegend()

# # QC 筛选
# rna <- subset(rna, subset = 
#                 nFeature_RNA > 350 & nFeature_RNA < 6000 &
#                 nCount_RNA > 300 & nCount_RNA < 40000 &
#                 percent.mt < 20)

# # 重新绘制 QC 指标图
# VlnPlot(rna, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
#         pt.size = 0.001, ncol = 3) + NoLegend()

# # 输出对象信息
# rna
# #=============================RNA=========================================


# #=============================ADT=========================================
# # ADT
# # load data
# adt_counts <- read.table(file = adt_path, sep = "\t", header = TRUE)

# # 创建 Seurat 对象
# adt <- CreateSeuratObject(counts = adt_counts)

# # 添加 nCount_adt 到元数据
# adt[["nCount_adt"]] <- Matrix::colSums(adt_counts)

# # 绘制 ADT 的 QC 小提琴图
# VlnPlot(adt, features = "nCount_adt", 
#         pt.size = 0.001, ncol = 1, log = TRUE) + NoLegend()

# # QC 筛选
# adt <- subset(adt, subset = nCount_adt > 500 & nCount_adt < 15000)

# # 绘制 QC 筛选后的 ADT 小提琴图
# VlnPlot(adt, features = "nCount_adt", 
#         pt.size = 0.001, ncol = 1, log = TRUE) + NoLegend()

# # 输出对象信息
# adt
# #=============================ADT=========================================

# #=============================HTO=========================================
# #筛选双细胞
# # HTO
# # load data
# hto_counts <- read.table(file = hto_path, sep = "\t", header = TRUE)
# hto_counts <- hto_counts[, colnames(rna_counts)]
# hto <- CreateSeuratObject(counts = hto_counts, assay = "hto")
# # remove doublets
# hto <- NormalizeData(hto, assay = "hto", normalization.method = "CLR", margin = 1)
# hto <- HTODemux(hto, assay = "hto", positive.quantile = 0.99)
# table(hto$hto_classification.global)
# Idents(hto) <- "hto_classification.global"
# hto <- subset(hto, idents = "Singlet")
# hto
# # Get intersected cells satisfying QC metrics of all modalities
# cell_ids <- Reduce(intersect, list(colnames(rna), colnames(adt), colnames(hto)))
# rna <- subset(rna, cells = cell_ids)
# adt <- subset(adt, cells = cell_ids)
# rna
# adt
# SaveH5Seurat(rna, file = "rna_data.h5Seurat")
# #=============================HTO=========================================


# # 保存 ADT 对象为 .h5Seurat 文件
# SaveH5Seurat(adt, file = "adt_data.h5Seurat")

# # RNA
# # 获取 RNA 计数矩阵（从 layers$counts）
# rna_counts <- rna@assays$RNA@layers$counts
# rna_meta <- rna@meta.data  # 获取 RNA 的元数据
# rna_active_ident <- rna@active.ident  # 获取 RNA 的 active.ident

# # 保存为 CSV 文件
# write.table(as.matrix(rna_counts), file = "rna_counts.csv", sep = ",", row.names = TRUE, col.names = NA)
# write.table(rna_meta, file = "rna_meta.csv", sep = ",", row.names = TRUE)
# write.table(as.data.frame(rna_active_ident), file = "rna_active_ident.csv", sep = ",", row.names = TRUE)

# # 获取 ADT 计数矩阵（稀疏矩阵）
# adt_counts <- adt@assays$RNA@layers$counts

# # 将稀疏矩阵转换为密集矩阵
# adt_counts_dense <- as.matrix(adt_counts)

# # 获取 ADT 的元数据
# adt_meta <- adt@meta.data

# # 获取 ADT 的 active.ident
# adt_active_ident <- adt@active.ident

# # 保存为 CSV 文件
# write.table(adt_counts_dense, file = "adt_counts.csv", sep = ",", row.names = TRUE, col.names = NA)
# write.table(adt_meta, file = "adt_meta.csv", sep = ",", row.names = TRUE)
# write.table(as.data.frame(adt_active_ident), file = "adt_active_ident.csv", sep = ",", row.names = TRUE)



# # 
# # # 保存 RNA 的稀疏矩阵
# # rna_counts_sparse <- as(rna_counts, "CsparseMatrix")
# # adt_counts_sparse <- as(adt_counts, "CsparseMatrix")
# # # 保存 RNA 的稀疏矩阵为 HDF5 格式
# # h5write(rna_counts_sparse, "rna_counts_sparse.h5", "counts")
# # 
# # # 保存 ADT 的稀疏矩阵为 HDF5 格式
# # h5write(adt_counts_sparse, "adt_counts_sparse.h5", "counts")
# # 
# # # 保存元数据为 HDF5 格式（可选）
# # h5write(rna_meta, "rna_meta.h5", "meta")
# # h5write(adt_meta, "adt_meta.h5", "meta")
