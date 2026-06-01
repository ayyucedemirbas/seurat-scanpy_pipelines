import os
import urllib.request
import tarfile
import numpy as np
import psutil

from scAnalysis import (
    sc_io,
    preprocessing,
    quality_control,
    cell_cycle,
    batch_correction,
    dimensionality,
    clustering,
    trajectory,
    differential,
    enrichment,
    visualization,
    interactive_viz,
    imputation,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

def print_memory_usage(step_name: str):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[Memory Usage] {step_name}: {mem_mb:.2f} MB")

def get_pbmc3k_data():
    url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    filepath = "pbmc3k.tar.gz"
    extract_path = "pbmc3k_extracted"

    if not os.path.exists(extract_path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(filepath, "wb") as out_file:
            out_file.write(response.read())
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=extract_path)

    return os.path.join(extract_path, "filtered_gene_bc_matrices", "hg19")


print_memory_usage("Before downloading/loading data")

data_path = get_pbmc3k_data()
data = sc_io.read_10x_mtx(data_path)
data.var.index = sc_io._make_unique(data.var.index.values)
print(f"Loaded {data.n_obs} cells and {data.n_vars} genes.")

print_memory_usage("After loading raw data")

preprocessing.calculate_qc_metrics(data, qc_vars=["MT-"])

quality_control.scrublet(data)

mask_singlets = ~data.obs['predicted_doublet'].astype(bool)
data = data[mask_singlets, :]

data = preprocessing.filter_cells(data, min_genes=200, max_pct_mito=5.0)
data = preprocessing.filter_genes(data, min_cells=3)

print_memory_usage("After QC and filtering")

# 3. Normalization (Choose ONE of the following methods)

# Option A: Basic total count normalization (Fastest, traditional approach)
preprocessing.normalize_total(data, target_sum=1e4)

# Option B: scran-like pooling (Recommended for shallow sequencing with many dropouts)
# preprocessing.normalize_scran_pooling(data, target_sum=1e4)

# Option C: sctransform-like regression (Advanced variance stabilization, regresses out sequencing depth)
# Note: If you use sctransform, you typically do NOT need to apply log1p afterwards,
# as the output is already Pearson residuals.
#preprocessing.normalize_sctransform(data)


# Apply log1p ONLY IF you used Option A or Option B above.
# (Comment this out if you used Option C - sctransform)
preprocessing.log1p(data)

print_memory_usage("After normalization and log1p")

#Weighted Neighborhood Imputation with Dropout Detection (WNID)
imputation.impute_wnid(data, k=3, dropout_thresh=0.9, n_pcs=30)

print_memory_usage("After WNID imputation")

# 4. Cell Cycle Scoring & Feature Selection
cell_cycle.score_cell_cycle(data, organism="human")

preprocessing.highly_variable_genes(data, n_top_genes=2000)

data.raw = data.copy()

preprocessing.scale(data, max_value=10)

data.obs['batch'] = np.random.choice(['Donor_A', 'Donor_B'], size=data.n_obs)

print_memory_usage("After scaling and setting up batch data")

dimensionality.run_pca(data, n_components=50)
batch_correction.harmony_integrate(data, batch_key='batch', basis='X_pca', adjusted_basis='X_pca_harmony')

print_memory_usage("After PCA and Harmony integration")

dimensionality.neighbors(data, n_neighbors=10, n_pcs=40)
dimensionality.run_umap(data, min_dist=0.3)

clustering.cluster_leiden(data, resolution=0.5, key_added="leiden")

print_memory_usage("After Neighbors, UMAP, and Leiden clustering")

dimensionality.run_diffmap(data)

root_idx = trajectory.select_root_cell(data, cluster_key='leiden', root_cluster='0', strategy='extreme')
trajectory.diffusion_pseudotime(data, root_cell=root_idx)

print_memory_usage("After Diffusion Pseudotime (DPT)")

differential.rank_genes_groups(data, groupby='leiden', method='t-test', use_raw=True)
cluster0_markers = differential.get_marker_genes(data, group='0', pval_cutoff=0.05, lfc_cutoff=0.5)
print(f"Top 3 significant markers for Cluster 0: {cluster0_markers.head(3)}")

canonical_markers = {
    'T_cell': ['CD3D', 'CD3E', 'CD3G'],
    'B_cell': ['CD19', 'MS4A1', 'CD79A'],
    'Monocyte': ['CD14', 'LYZ', 'S100A8']
}

enrichment_results = enrichment.rank_genes_groups_by_enrichment(data, canonical_markers, groupby='leiden')

print_memory_usage("After differential expression and scoring")

visualization.plot_umap(data, color="leiden", title="PBMC 3k - Leiden Clusters", save="umap_clusters.png")

visualization.plot_umap(data, color="phase", title="Cell Cycle Phase Distribution", save="umap_phase.png")

visualization.volcano_plot(data, group='0', top_n_genes=5, save="volcano_cluster0.png")

available_markers = [g for g in ['CD3E', 'MS4A1', 'CD14', 'LYZ', 'GNLY'] if g in data.var.index]
if available_markers:
    visualization.plot_dotplot(data, var_names=available_markers, groupby='leiden', save="dotplot_markers.png")

try:
    interactive_viz.interactive_embedding(
        data,
        basis='X_umap',
        color='leiden',
        hover_data=['n_genes_by_counts', 'phase', 'dpt_pseudotime'],
        title="PBMC 3k - Interactive UMAP",
        save_html="interactive_umap.html"
    )
except Exception as e:
    print(f"Skipping interactive plots (Plotly may not be installed): {e}")

try:
    interactive_viz.interactive_3d_embedding(
        data,
        basis='X_pca',
        color='leiden',
        dimensions=[0, 1, 2],
        save_html="interactive_3d_pca.html"
    )

    interactive_viz.interactive_violin(
        data,
        keys=['n_genes_by_counts', 'total_counts', 'dpt_pseudotime'],
        groupby='leiden',
        save_html="interactive_violin.html"
    )
    if available_markers:
        interactive_viz.interactive_heatmap(
            data,
            var_names=available_markers,
            groupby='leiden',
            use_raw=True,
            save_html="interactive_heatmap.html"
        )

    print("Interactive plots saved successfully.")

except Exception as e:
    print(f"Skipping interactive plots (Plotly may not be installed or an error occurred): {e}")

print_memory_usage("Before saving H5AD file")

output_file = "pbmc3k_full_analysis.h5ad"
sc_io.write_h5ad(data, output_file)

print_memory_usage("End of script")
