library(Seurat)
library(Signac)  
library(igraph)
library(aricode)

data_name = "PBMC_TEA"

RNA_path <- file.path("data", data_name, "Processed_dataset/RNA")  
ATAC_path <- file.path("data", data_name, "Processed_dataset/ATAC")
WNN_path <- file.path("data", data_name, "graph", paste0(data_name, "_wnn_edgelist_k15.tsv"))

rna_data <- Read10X(data.dir = RNA_path,gene.column = 1)
seurat_obj <- CreateSeuratObject(counts = rna_data, assay = "RNA")
atac_data <- Read10X(data.dir = ATAC_path,gene.column = 1)
chrom_assay <- CreateChromatinAssay(counts = atac_data,sep = c("-", "-"))
seurat_obj[["ATAC"]] <- chrom_assay

seurat_obj <- SCTransform(seurat_obj,  assay = "RNA", verbose = FALSE)
seurat_obj <- RunPCA(seurat_obj, assay = "SCT", npcs = 50)

DefaultAssay(seurat_obj) <- "ATAC"
seurat_obj <- RunTFIDF(seurat_obj) 
seurat_obj <- FindTopFeatures(seurat_obj, min.cutoff = "q0")  
seurat_obj <- RunSVD(seurat_obj) 

seurat_obj <- FindMultiModalNeighbors(
object = seurat_obj,
reduction.list = list("pca", "lsi"),  
dims.list = list(1:50, 2:50), 
k.nn = 15)

wnn_graph <- seurat_obj@graphs$wsnn
wnn_edgelist <- as.data.frame(as_edgelist(graph_from_adjacency_matrix(wnn_graph, weighted = TRUE)))
colnames(wnn_edgelist) <- NULL 

write.table(wnn_edgelist, WNN_path, sep = "\t", row.names = FALSE)
