import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from time import time
import warnings
warnings.filterwarnings("ignore")
def get_anndata(data_now, Batch, all_label= False, all_batch = False):
    if all_label:
        data_now = data_now.loc[data_now['celltype'] != 'unknown_cell']
    if not all_batch:
        data_now = data_now.loc[data_now['batch'] == Batch]
    X = np.array(data_now.drop(['batch', 'celltype'],axis=1))
    obs = data_now[['batch', 'celltype']]
    var = pd.DataFrame(data_now.drop(['batch', 'celltype'],axis=1).columns.tolist() )
    var.columns = ['Protein name']
    var.index = data_now.drop(['batch', 'celltype'],axis=1).columns.tolist()
    adata = ad.AnnData(X = X, obs = obs, var = var)
    return adata

results_file = './write/scanpy_result_all_cell.h5ad'



print('---------------------------\n loading data')
t0 = time()
cell_label_info = pd.read_table('../data/cell_label.txt')
data = pd.read_csv('../processd_data.csv')
data_label = pd.read_table('../data/data.txt')
print('data loaded, it takes %f seconds \n'%(time() - t0))


cell_label_list = cell_label_info.columns.tolist()
cell_label_list.append('unknown_cell')
data_just_maker = data[data_label.columns]
marker_genes = data_just_maker.drop(['batch', 'celltype'],axis=1).columns.tolist()
gene_list = data.drop(['batch', 'celltype','celltype_number'],axis=1).columns.tolist()


adata_allcell = get_anndata(data_just_maker, Batch = '', all_label=False, all_batch=True)

print('building the neighbors graph')
t0 = time()
sc.pp.neighbors(adata_allcell, n_neighbors=15)
print('graph built, it takes %f seconds \n'%(time() - t0))

print('Embedding the Umap')
t0 = time()
sc.tl.umap(adata_allcell)
print('Embedding finished, it takes %f seconds  \n'%(time() - t0))

print('Start clustering')
t0 = time()
sc.tl.louvain(adata_allcell)
print('Get the cluster, it takes %f seconds  \n'%(time() - t0))

sc.pl.umap(adata_allcell, color=['louvain'], legend_loc='on data', save='_justmaker_louvain.pdf')

sc.pl.umap(adata_allcell, color=marker_genes, save='_justmaker_makergenes.pdf')

sc.tl.rank_genes_groups(adata_allcell, 'louvain', method='t-test')
sc.pl.rank_genes_groups(adata_allcell, sharey=True, save='_t-test.pdf')

ax = sc.pl.dotplot(adata_allcell, marker_genes, groupby='louvain', save = '.pdf')

ax = sc.pl.stacked_violin(adata_allcell, marker_genes, groupby='louvain', rotation=90, save = '.pdf')

adata_allcell.write(results_file)

print("start plot the headmap")
t0 = time()
ax = sc.pl.heatmap(adata_allcell, marker_genes, groupby='louvain',cmap='bwr', figsize=(5, 8), save = '.pdf')
print('headmap finished, it takes %f seconds  \n'%(time() - t0))
