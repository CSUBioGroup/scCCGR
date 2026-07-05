import torch
from network import Network
from metric import *
import numpy as np
import argparse
import random
import dataloader as loader
import os
import torch.nn.functional as F
from loss import *


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    g = torch.Generator()
    g.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

def contrastive_train_epoch(model, 
                            adjs, 
                            adjs_n, 
                            g_feats, 
                            recon_para, 
                            optimizer, 
                            threshold, 
                            view):
    model.train()
    rec_cri = ReconsLoss()

    hs, qs, xrs, _ = model( adjs_n = adjs_n, g_feats = g_feats)   
    
    loss_list = []
    pse = [] 

    w = view-1
    for v in range(view): 
        sim = torch.exp(torch.mm(hs[v], hs[w].t()))
        sim_probs = sim / sim.sum(1, keepdim=True)

        Q = torch.mm(qs[v], qs[w].t())
        Q.fill_diagonal_(1)
        pos_mask = (Q >= threshold).float()
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdims=True)

        if v == w:
            Q_gloal = Q

        pse.append(Q)
        loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()

        loss_list.append(loss_contrast)

        loss_list.append(rec_cri(xrs[v], adjs[v], recon_para[v]))

    for v in range(view - 1):
        loss_each = F.mse_loss(Q_gloal, pse[v])
        loss_list.append(loss_each)
    
    optimizer.zero_grad()
    loss = sum(loss_list)
    loss.backward()
    optimizer.step()

    return loss.item()

def main():  
    parser = argparse.ArgumentParser(description='train')

    Dataname = 'PBMC_TEA'
    parser.add_argument('--dataset', default=Dataname)

    parser.add_argument("--mid_layer_dims", type=list, default=[48, 48, 48, 48])
    parser.add_argument("--feature_dim", type=int, default=32) 
    parser.add_argument("--high_feature_dim", type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=342)

    parser.add_argument("--con_lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--con_epochs", type=int, default=290)
    parser.add_argument("--valid_interval", type=int, default=1000)
    parser.add_argument("--save", type=bool, default=False)

    args = parser.parse_args()

    setup_seed(args.seed)

    device = torch.device('cuda:3')
    g_feats, adjs_n, adjs, Y = loader.load_moco_data(args.dataset)
    joint_dim = g_feats[0].shape[1]

    dims = [g_feats[0].shape[1], g_feats[1]. shape[1], g_feats[2]. shape[1], joint_dim]
    view = 4
    data_size = g_feats[0].shape[0]
    class_num = len(np.unique(Y))

    print(f"Dataset: {args.dataset}, Dims:{dims}, Views: {view}, Data Size: {data_size}, Class Num: {class_num}")

    model = Network(view = view, 
                    input_dims = dims, 
                    mid_layer_dims = args.mid_layer_dims, 
                    feature_dim = args.feature_dim,  
                    high_feature_dim = args.high_feature_dim, 
                    class_num = class_num, 
                    dropout = args.dropout)

    model = model.to(device)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.con_lr, weight_decay=args.weight_decay)

    epoch = 1

    recon_para = []
    for v in range(view):
        num_sample = adjs[v].shape[0]
        pos_weight = float(num_sample * num_sample - adjs[v].sum()) / adjs[v].sum()
        norm = num_sample * num_sample / float((num_sample * num_sample - adjs[v].sum()) * 2)
        recon_para.append([norm, pos_weight])

    for v in range(view):
        adjs[v] = torch.as_tensor(adjs[v], dtype=torch.float32).to(device)
        adjs_n[v] = adjs_n[v].to(device)
    
    for v in range(view - 1):
        g_feats[v] = torch.as_tensor(g_feats[v].copy(), dtype=torch.float32).to(device)

    while epoch <= args.con_epochs:

        loss = contrastive_train_epoch(model = model, 
                                       adjs = adjs, 
                                       adjs_n = adjs_n, 
                                       g_feats = g_feats, 
                                       recon_para = recon_para, 
                                       optimizer = optimizer2, 
                                       threshold = args.threshold, 
                                       view = view)

        if(epoch % args.valid_interval == 0):
            NMI, ARI = con_valid(model=model, adjs_n=adjs_n, g_feats=g_feats, data_size=data_size, 
                                 class_num=class_num, y=Y, save=False, dataset_name=args.dataset)

        epoch += 1

    NMI, ARI = con_valid(model=model, adjs_n=adjs_n, g_feats=g_feats, data_size=data_size, 
                         class_num=class_num, y=Y, save=args.save, dataset_name=args.dataset)

if __name__ == "__main__":
    main()
