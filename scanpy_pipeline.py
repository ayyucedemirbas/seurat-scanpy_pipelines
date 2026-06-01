#!pip install scanpy anndata scrublet harmonypy leidenalg plotly wget matplotlib seaborn pandas numpy

import os
import tarfile
import urllib.request
import shutil
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import harmonypy as hm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import psutil

sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, facecolor='white')

def print_memory_usage(step_name: str):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[Memory Usage] {step_name}: {mem_mb:.2f} MB")

def impute_wnid_scanpy(adata: sc.AnnData, k: int = 3, dropout_thresh: float = 0.9, n_pcs: int = 30, random_state: int = 0):
    X = adata.X
    is_sparse = sp.issparse(X)

    if is_sparse:
        X_dense = X.toarray()
    else:
        X_dense = X.copy()

    n_comp = min(n_pcs, X_dense.shape[1] - 1)
    pca = PCA(n_components=n_comp, random_state=random_state)
    pca_emb = pca.fit_transform(X_dense)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nbrs.fit(pca_emb)
    distances, indices = nbrs.kneighbors(pca_emb)

    gene_means = X_dense.mean(axis=0)
    gene_vars = X_dense.var(axis=0)

    mu = np.where(gene_means == 0, 1e-12, gene_means)
    dispersion = gene_vars / mu

    dropout_prob = np.exp(-mu / np.maximum(dispersion, 1e-12))
    dropout_mask = (X_dense == 0) & (dropout_prob > dropout_thresh)

    X_imputed = X_dense.copy()

    for i in range(X_dense.shape[0]):
        cell_dropouts = dropout_mask[i]

        if not np.any(cell_dropouts):
            continue

        neighbor_idx = indices[i, 1:]
        dists = distances[i, 1:]

        weights = np.exp(-dists)
        weights_sum = weights.sum()

        if weights_sum == 0:
            continue

        weights /= weights_sum

        neighbor_expr = X_dense[neighbor_idx][:, cell_dropouts]
        imputed_values = np.dot(weights, neighbor_expr)

        X_imputed[i, cell_dropouts] = imputed_values


    if is_sparse:
        adata.X = sp.csr_matrix(X_imputed)
    else:
        adata.X = X_imputed


print_memory_usage("Before downloading/loading data")

url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
filepath = "pbmc3k.tar.gz"
extract_path = "pbmc3k_extracted"
data_dir = os.path.join(extract_path, "filtered_gene_bc_matrices", "hg19")

if not os.path.exists(data_dir):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(filepath, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_path)


adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
adata.var_names_make_unique()

print_memory_usage("After loading raw data")

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.external.pp.scrublet(adata)
adata = adata[~adata.obs['predicted_doublet'], :].copy()

sc.pp.filter_cells(adata, min_genes=200)
adata = adata[adata.obs['n_genes_by_counts'] < 2500, :]
adata = adata[adata.obs['pct_counts_mt'] < 5, :]
sc.pp.filter_genes(adata, min_cells=3)

print_memory_usage("After QC and filtering")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print_memory_usage("After normalization and log1p")

impute_wnid_scanpy(adata, k=3, dropout_thresh=0.9, n_pcs=30)

print_memory_usage("After WNID imputation")

s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6']
g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2']
s_genes = [g for g in s_genes if g in adata.var_names]
g2m_genes = [g for g in g2m_genes if g in adata.var_names]
sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata.raw = adata

sc.pp.scale(adata, max_value=10)
adata.obs['batch'] = pd.Categorical(np.random.choice(['Donor_A', 'Donor_B'], size=adata.n_obs))

print_memory_usage("After scaling and setting up batch data")

sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, ['batch'])
if ho.Z_corr.shape[0] == adata.n_obs:
    adata.obsm['X_pca_harmony'] = ho.Z_corr
else:
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T

print_memory_usage("After PCA and Harmony integration")

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep='X_pca_harmony')
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')

print_memory_usage("After Neighbors, UMAP, and Leiden clustering")

sc.tl.diffmap(adata)
root_idx = np.where(adata.obs['leiden'] == '0')[0][0]
adata.uns['iroot'] = root_idx
sc.tl.dpt(adata)

print_memory_usage("After Diffusion Pseudotime (DPT)")

sc.tl.rank_genes_groups(adata, groupby='leiden', method='t-test', use_raw=True)

canonical_markers = ['CD3D', 'CD3E', 'CD3G', 'CD19', 'MS4A1', 'CD14', 'LYZ']
available_markers = [g for g in canonical_markers if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=[g for g in ['CD3D', 'CD3E'] if g in adata.var_names], score_name='T_cell_score')

print_memory_usage("After differential expression and scoring")

sc.pl.umap(adata, color=['leiden', 'batch'], save='_clusters.png', show=False)
sc.pl.umap(adata, color='phase', save='_phase.png', show=False)

if available_markers:
    sc.pl.dotplot(adata, var_names=available_markers, groupby='leiden', standard_scale='var', save='_markers.png', show=False)

result = adata.uns['rank_genes_groups']
df_volcano = pd.DataFrame({
    'gene': result['names']['0'],
    'lfc': result['logfoldchanges']['0'],
    'pval_adj': result['pvals_adj']['0']
})

df_volcano['nlog10_pval'] = -np.log10(df_volcano['pval_adj'].clip(lower=1e-300))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_volcano, x='lfc', y='nlog10_pval', color="grey", alpha=0.5, edgecolor=None, s=15)
sig = df_volcano[(df_volcano['lfc'] > 0.5) & (df_volcano['pval_adj'] < 0.05)]
sns.scatterplot(data=sig, x='lfc', y='nlog10_pval', color="red", alpha=0.8, edgecolor=None, s=20)

plt.axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)
plt.axvline(x=-0.5, color='black', linestyle='--', linewidth=0.5)
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=0.5)
plt.title("Volcano Plot - Cluster 0")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 Adjusted P-value")
plt.savefig("volcano_cluster0.png", bbox_inches='tight', dpi=100)
plt.close()

df_plot = pd.DataFrame({
    'UMAP1': adata.obsm['X_umap'][:, 0],
    'UMAP2': adata.obsm['X_umap'][:, 1],
    'Cluster': adata.obs['leiden'],
    'Pseudotime': adata.obs['dpt_pseudotime']
})
fig = px.scatter(df_plot, x='UMAP1', y='UMAP2', color='Cluster', hover_data=['Pseudotime'], title='Interactive UMAP')
fig.write_html("interactive_umap.html")

sc.tl.rank_genes_groups(adata, groupby='leiden', method='t-test', use_raw=True)

cluster0_top_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])['0'].head(3).tolist()
print(f"Top 3 markers for Cluster 0: {cluster0_top_genes}")

print_memory_usage("Before saving H5AD file")

adata.write("pbmc3k_full_analysis_scanpy.h5ad")

print_memory_usage("End of script")
