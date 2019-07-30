import scanpy as sc
import pandas as pd
import numpy as np
from time import time
import anndata as ad
import warnings
warnings.filterwarnings("ignore")


# convert the data to anndata objet
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


def single_batch(batch_info):
    results_file = '../write/single_batch_' + batch_info + '.h5ad'
    adata_justmarker = get_anndata(data_just_maker, Batch=batch_info, all_label=False, all_batch=False)

    print('building the neighbors graph')
    t0 = time()
    sc.pp.neighbors(adata_justmarker, n_neighbors=15)
    print('graph built, it takes %f seconds \n' % (time() - t0))

    print('Embedding the Umap')
    t0 = time()
    sc.tl.umap(adata_justmarker)
    print('Embedding finished, it takes %f seconds  \n' % (time() - t0))

    print('Start clustering')
    t0 = time()
    sc.tl.louvain(adata_justmarker)
    print('Get the cluster, it takes %f seconds  \n' % (time() - t0))

    sc.pl.umap(adata_justmarker, color=['louvain'], legend_loc='on data', save='_' +
                                                                               batch_info+'_justmaker_louvain.pdf')

    sc.pl.umap(adata_justmarker, color=marker_genes, save='_' +
                                                                               batch_info+'_justmaker_makergenes.pdf')

    sc.tl.rank_genes_groups(adata_justmarker, 'louvain', method='t-test')
    sc.pl.rank_genes_groups(adata_justmarker, sharey=True, save='_' +
                                                                               batch_info+'_t-test.pdf')

    ax = sc.pl.dotplot(adata_justmarker, marker_genes, groupby='louvain', save='_' +
                                                                               batch_info+'.pdf')

    ax = sc.pl.stacked_violin(adata_justmarker, marker_genes, groupby='louvain', rotation=90, save='_' +
                                                                               batch_info+'.pdf')

    adata_justmarker.write(results_file)
    print("start plot the headmap")
    t0 = time()
    ax = sc.pl.heatmap(adata_justmarker, marker_genes, groupby='louvain', cmap='bwr', figsize=(5, 8), save='_' +
                                                                               batch_info+'.pdf')
    print('headmap finished, it takes %f seconds  \n' % (time() - t0))


if __name__ == '__main__':
    print("Start!")
    # set the result path
    results_file = '../write/'

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

    for batch in list(set(data_just_maker['batch'])):
        print(batch, '  start')
        t0 = time()
        single_batch(batch)
        print(batch, '  finished')
