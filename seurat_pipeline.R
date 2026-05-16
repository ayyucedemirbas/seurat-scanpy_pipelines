library(Seurat)

options(Seurat.object.assay.version = "v3")


library(DoubletFinder)
library(harmony)
library(ggplot2)
library(plotly)

url <- "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
filepath <- "pbmc3k.tar.gz"
extract_path <- "pbmc3k_extracted"
data_dir <- file.path(extract_path, "filtered_gene_bc_matrices", "hg19")

if (!dir.exists(data_dir)) {
  download.file(url, filepath)
  untar(filepath, exdir = extract_path)
}

counts <- Read10X(data.dir = data_dir)
pbmc <- CreateSeuratObject(counts = counts, project = "PBMC3k", min.cells = 3, min.features = 200)

pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
pbmc <- subset(pbmc, subset = nFeature_RNA < 2500 & percent.mt < 5)

pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
pbmc <- ScaleData(pbmc)
pbmc <- RunPCA(pbmc, npcs = 50, verbose = FALSE)
pbmc <- RunUMAP(pbmc, dims = 1:40, verbose = FALSE)

sweep.res.list <- paramSweep(pbmc, PCs = 1:10, sct = FALSE)
sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
bcmvn <- find.pK(sweep.stats)
pK_val <- as.numeric(as.character(bcmvn$pK[which.max(bcmvn$BCmetric)]))
nExp_poi <- round(0.06 * nrow(pbmc@meta.data)) # Assuming ~6% doublet rate


pbmc <- doubletFinder(pbmc, PCs = 1:10, pN = 0.25, pK = pK_val, nExp = nExp_poi, reuse.pANN = NULL, sct = FALSE)


df_col <- grep("DF.classifications", colnames(pbmc@meta.data), value = TRUE)
pbmc <- subset(pbmc, cells = rownames(pbmc@meta.data)[pbmc@meta.data[[df_col]] == "Singlet"])

s.genes <- cc.genes$s.genes
g2m.genes <- cc.genes$g2m.genes
pbmc <- CellCycleScoring(pbmc, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)

pbmc$batch <- sample(c("Donor_A", "Donor_B"), size = ncol(pbmc), replace = TRUE)
pbmc <- RunHarmony(pbmc, group.by.vars = "batch", plot_convergence = FALSE)

pbmc <- FindNeighbors(pbmc, reduction = "harmony", dims = 1:40)
pbmc <- FindClusters(pbmc, resolution = 0.5, algorithm = 1) 

# Seurat does not have built-in Diffusion Pseudotime. 
pbmc$pseudotime_mock <- runif(ncol(pbmc)) # Placeholder for pseudotime

markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.5, test.use = "t")
cluster0_markers <- head(subset(markers, cluster == 0), 3)
print("Top 3 markers for Cluster 0:")
print(cluster0_markers)

canonical_markers <- list(T_cell = c('CD3D', 'CD3E', 'CD3G'), B_cell = c('CD19', 'MS4A1'))
pbmc <- AddModuleScore(pbmc, features = canonical_markers, name = "MarkerScore")

png("umap_clusters_seurat.png", width=800, height=600)
print(DimPlot(pbmc, reduction = "umap", group.by = "seurat_clusters", label = TRUE))
dev.off()

png("umap_phase_seurat.png", width=800, height=600)
print(DimPlot(pbmc, reduction = "umap", group.by = "Phase"))
dev.off()

avail_markers <- intersect(c('CD3E', 'MS4A1', 'CD14', 'LYZ', 'GNLY'), rownames(pbmc))
if (length(avail_markers) > 0) {
  png("dotplot_markers_seurat.png", width=800, height=600)
  print(DotPlot(pbmc, features = avail_markers) + theme(axis.text.x = element_text(angle = 45, hjust=1)))
  dev.off()
}

plot_data <- data.frame(
  UMAP1 = Embeddings(pbmc, "umap")[, 1],
  UMAP2 = Embeddings(pbmc, "umap")[, 2],
  Cluster = pbmc$seurat_clusters
)
fig <- plot_ly(plot_data, x = ~UMAP1, y = ~UMAP2, color = ~Cluster, type = 'scatter', mode = 'markers',
               marker = list(size = 3)) %>%
  layout(title = "Interactive UMAP - PBMC 3k")
htmlwidgets::saveWidget(fig, "interactive_umap_seurat.html")

saveRDS(pbmc, file = "pbmc3k_full_analysis_seurat.rds")
