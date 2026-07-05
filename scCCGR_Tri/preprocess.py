import scanpy as sc
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance
from scipy import sparse as sp
import torch
import pandas as pd
import anndata as ad
import hnswlib
from muon import prot as pt


def normalizeRNA(adata, n_top_genes = 2000):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    sc.pp.highly_variable_genes(adata,n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack',n_comps=50)      
    return adata.obsm['X_pca']


def normalizeATAC(adata, n_top_genes = 30000):
    sc.pp.filter_genes(adata, min_cells=3)
    
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(adata.X)

    svd = TruncatedSVD(n_components=50)  # 选择50个组分
    lsi_components = svd.fit_transform(tfidf_matrix)

    return  lsi_components

def normalizeADT(adata):
    adata.X = adata.X.tocsc()
    pt.pp.clr(adata)  #normalize

    if(adata.n_vars > 50):
        sc.pp.pca(adata, n_comps=50)
        X = adata.obsm["X_pca"]
    else:
        X = adata.X.toarray()

    return X

def cal_nn(x, k=500, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements=max_element, 
                 ef_construction=600, 
                 random_seed=600,
                 M=100)
    
    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = k)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    return neighbors, distance 

def build_knn_graph(z,n_neighbors=15):
    adata = ad.AnnData(z)
    sc.pp.neighbors(adata, use_rep = 'X', n_neighbors = 15)
    conn = adata.obsp['connectivities']
    binary_conn = conn.copy()
    binary_conn.data = np.ones_like(conn.data)
    return binary_conn.tocoo()
   
def sys_normalized_adjacency(adj):   
   row_sum = np.array(adj.sum(1))
   row_sum = (row_sum == 0)*1 + row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def adj_normalize(adj):
    adj_normalized = sys_normalized_adjacency(adj + sp.eye(adj.shape[0]))  
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return adj_normalized

def build_adjacency_from_edgelist(barcode_file, edgelist_file):
    barcodes = pd.read_csv(barcode_file, sep='\t', header=None)[0].values
    num_cells = len(barcodes)
    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    
    edges_df = pd.read_csv(edgelist_file, sep='\t', header=None)
    edges = []
    
    for _, row in edges_df.iterrows():
        src, dst = row[0], row[1]
        if src in barcode_to_idx and dst in barcode_to_idx:
            edges.append([barcode_to_idx[src], barcode_to_idx[dst]])
    
    edges = np.array(edges)
    
    edges_inverse = edges[:, [1, 0]]
    all_edges = np.concatenate([edges, edges_inverse])
    all_edges = np.unique(all_edges, axis=0)  
    
    adj = sp.coo_matrix((np.ones(all_edges.shape[0]), 
                     (all_edges[:, 0], all_edges[:, 1])),
                    shape=(num_cells, num_cells), dtype=np.float32)
    
    return adj

   



