import pickle
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from preprocess import *

class GetData(Dataset):
    def __init__(self, dataset_name):
        label_path = 'data/' + dataset_name + '/' + dataset_name+'_cluster.csv'

        ATAC_matrix_path = 'data/' + dataset_name + '/Processed_dataset/ATAC/matrix.mtx'
        RNA_matrix_path = 'data/' + dataset_name + '/Processed_dataset/RNA/matrix.mtx'

        label = pd.read_csv(label_path, sep = ',', header=0)
        Y = np.array(label['cluster_id'])

        ATAC_matrix = sc.read_mtx(ATAC_matrix_path)
        RNA_matrix = sc.read_mtx(RNA_matrix_path)

        ATAC_matrix = ATAC_matrix.transpose()
        RNA_matrix = RNA_matrix.transpose()

        ATAC_matrix = normalizeATAC(ATAC_matrix, n_top_genes = ATAC_matrix.n_vars//2)
        RNA_matrix = normalizeRNA(RNA_matrix, n_top_genes = 2000)

        ATAC_neighbors, _ = cal_nn(ATAC_matrix,k=6)
        RNA_neighbors, _ = cal_nn(RNA_matrix,k=6)

        self.g_feats = [ATAC_matrix, RNA_matrix]
        self.neighbors = [ATAC_neighbors, RNA_neighbors]

        self.Y = Y
        self.V1 = ATAC_matrix
        self.V2 = RNA_matrix

def load_data(dataset_name):
    dataset = GetData(dataset_name)
    dims = [dataset.V1.shape[1], dataset.V2.shape[1]]
    view = 2
    data_size = dataset.V1.shape[0]
    class_num = len(np.unique(dataset.Y))

    return dataset, dims, view, data_size, class_num

def load_moco_data(dataset_name):
    barcode_path = 'data/' + dataset_name + '/Processed_dataset/RNA/barcodes.tsv'
    wnn_edgelist_path = 'data/' + dataset_name + '/graph/' + dataset_name + '_wnn_edgelist_k15.tsv'
    label_path = 'data/' + dataset_name + '/' + dataset_name+'_cluster.csv'

    feat_data = pickle.load(open(os.path.join('lowdimention', f"{dataset_name}_moco_embedding.pkl"), "rb"))
    z_atac = feat_data['z0']
    z_rna = feat_data['z1']

    atac_adj = build_knn_graph(z_atac)
    rna_adj = build_knn_graph(z_rna)
    wnn_adj = build_adjacency_from_edgelist(barcode_path,wnn_edgelist_path)

    atac_adj_n = adj_normalize(atac_adj)
    rna_adj_n = adj_normalize(rna_adj)
    wnn_adj_n = adj_normalize(wnn_adj)

    adjs = [atac_adj.toarray(), rna_adj.toarray(), wnn_adj.toarray()]
    adjs_n = [atac_adj_n, rna_adj_n, wnn_adj_n]
    g_feats = [z_atac, z_rna]

    label = pd.read_csv(label_path, sep = ',', header=0)
    Y = np.array(label['cluster_id'])

    return g_feats, adjs_n, adjs, Y


