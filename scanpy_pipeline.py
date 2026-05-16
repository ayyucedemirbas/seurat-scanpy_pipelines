#!pip install scanpy anndata scrublet harmonypy leidenalg plotly wget matplotlib seaborn pandas numpy

import os
import tarfile
import urllib.request
import shutil
import scanpy as sc
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import harmonypy as hm

sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, facecolor='white')

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


adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


sc.external.pp.scrublet(adata)

adata = adata[~adata.obs['predicted_doublet'], :].copy()

sc.pp.filter_cells(adata, min_genes=200)
adata = adata[adata.obs['n_genes_by_counts'] < 2500, :]
adata = adata[adata.obs['pct_counts_mt'] < 5, :]
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

s_genes = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6']
g2m_genes = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2']
s_genes = [g for g in s_genes if g in adata.var_names]
g2m_genes = [g for g in g2m_genes if g in adata.var_names]
sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata.raw = adata # Save raw data
sc.pp.scale(adata, max_value=10)

adata.obs['batch'] = pd.Categorical(np.random.choice(['Donor_A', 'Donor_B'], size=adata.n_obs))
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)


ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, ['batch'])


if ho.Z_corr.shape[0] == adata.n_obs:
    adata.obsm['X_pca_harmony'] = ho.Z_corr
else:
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep='X_pca_harmony')
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')

sc.tl.diffmap(adata)
root_idx = np.where(adata.obs['leiden'] == '0')[0][0]
adata.uns['iroot'] = root_idx
sc.tl.dpt(adata)

sc.tl.rank_genes_groups(adata, groupby='leiden', method='t-test', use_raw=True)

canonical_markers = ['CD3D', 'CD3E', 'CD3G', 'CD19', 'MS4A1', 'CD14', 'LYZ']
available_markers = [g for g in canonical_markers if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=[g for g in ['CD3D', 'CD3E'] if g in adata.var_names], score_name='T_cell_score')

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
sns.scatterplot(data=df_volcano, x='lfc', y='nlog10_pval',
                color="grey", alpha=0.5, edgecolor=None, s=15)

sig = df_volcano[(df_volcano['lfc'] > 0.5) & (df_volcano['pval_adj'] < 0.05)]
sns.scatterplot(data=sig, x='lfc', y='nlog10_pval',
                color="red", alpha=0.8, edgecolor=None, s=20)

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

adata.write("pbmc3k_full_analysis_scanpy.h5ad")
