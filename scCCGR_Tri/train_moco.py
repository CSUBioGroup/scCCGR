import torch
from network import Encoder, MultiOmicsMoCo
from metric import *
import numpy as np
import argparse
import random
import dataloader as loader
import os
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


def pretrain_epoch(model, 
                   data, 
                   neighbors, 
                   batch_size, 
                   optimizer, 
                   device, 
                   view,
                   c=1):
    loss_epoch = 0.0
    loss1_epoch = 0.0
    loss2_epoch = 0.0
    loss3_epoch = 0.0

    indices = np.arange(data[0].shape[0])
    np.random.shuffle(indices)
    count = 0

    model.train()

    for step, pre_index in enumerate(range(data[0].shape[0] // batch_size + 1)):
        indices_idx = np.arange(pre_index * batch_size, min(data[0].shape[0], (pre_index + 1) * batch_size))
        if len(indices_idx) < batch_size:
            continue
        count += 1
        batch_indices = indices[indices_idx]  

        x = [torch.FloatTensor(modality[batch_indices]).to(device) for modality in data]     

        if neighbors is not None:
            batch_nei = [nei[batch_indices] for nei in neighbors]
            batch_nei_idx = [np.array([np.random.choice(n, c) for n in nei]) for nei in batch_nei]
            batch_nei_idx = [b.flatten() for b in batch_nei_idx]
            x_nei = [torch.FloatTensor(modality[idx]).to(device) for modality, idx in zip(data, batch_nei_idx)]
        
        for i in range(view):
            assert x_nei[i].size(0) // x[i].size(0) == c, \
                f"Modal {i}: neighbor batch size mismatch — got {x_nei[i].size(0)} vs {x[i].size(0)}"
        
        loss, loss1, loss2, loss3 = model(x1 = x[0], x2 = x[1], x3 = x[2], x1_aug = x_nei[0], x2_aug = x_nei[1], x3_aug = x_nei[2])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        loss1_epoch += loss1.item()
        loss2_epoch += loss2.item()
        loss3_epoch += loss3.item()
    
    loss_epoch = loss_epoch / count
    loss1_epoch = loss1_epoch / count
    loss2_epoch = loss2_epoch / count
    loss3_epoch = loss3_epoch / count

    return loss_epoch, loss1_epoch, loss2_epoch, loss3_epoch


def main():
    Dataname = 'PBMC_TEA'
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--dataset', default=Dataname)

    parser.add_argument('--moco_dims1', type=list, default=[64, 48, 32]),
    parser.add_argument('--moco_dims2', type=list, default=[64, 48, 32]),
    parser.add_argument('--moco_dims3', type=list, default=[64, 48, 32]),
    parser.add_argument("--K", type=float, default=2048), 
    parser.add_argument("--m", type=float, default=0.990) 
    parser.add_argument("--T", type=float, default=0.7) 
    parser.add_argument("--p", type=float, default=0.2) 
    parser.add_argument("--lam", type=float, default=0.2) 
    parser.add_argument("--alpha", type=float, default=0.1) 

    parser.add_argument('--seed', type=int, default=342)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--pre_lr", type=float, default=0.001)
    parser.add_argument("--pre_epochs", type=float, default=110)
    parser.add_argument("--c", type=float, default=1) 
    parser.add_argument("--valid_interval", type=float, default=1000)
    parser.add_argument("--save", type=bool, default=False)

    args = parser.parse_args()
    setup_seed(args.seed)


    device = torch.device('cuda:1')
    dataset, dims, view, data_size, class_num = loader.load_data(args.dataset)
    g_feats = dataset.g_feats
    neighbors = dataset.neighbors
    y = dataset.Y

    print(f"Dataset: {args.dataset}, dims:{dims}, Views: {view}, Data Size: {data_size}, Class Num: {class_num}")

    MOmoco = MultiOmicsMoCo(encoder = Encoder,
                            in_features1 = dims[0],
                            in_features2 = dims[1],
                            in_features3 = dims[2],
                            num_cluster= class_num,
                            latent_features1=args.moco_dims1,
                            latent_features2=args.moco_dims2,
                            latent_features3=args.moco_dims3,
                            K = args.K,
                            m = args.m,
                            T = args.T,
                            p = args.p,
                            lam = args.lam,
                            alpha= args.alpha)

    MOmoco = MOmoco.to(device)

    optimizer1 = torch.optim.Adam(MOmoco.parameters(), lr=args.pre_lr, weight_decay=0.0)

    epoch = 1

    while epoch <= args.pre_epochs:
        loss,loss1,loss2,loss3 = pretrain_epoch(model = MOmoco, 
                                          data = g_feats, 
                                          neighbors = neighbors, 
                                          batch_size = args.batch_size, 
                                          optimizer = optimizer1, 
                                          device = device, 
                                          view = view, 
                                          c=args.c)

        if(epoch % args.valid_interval == 0 ): 
            NMI, ARI = moco_valid(model=MOmoco, dataset_name = args.dataset, data_size=data_size, 
                                    class_num = class_num , g_feats = g_feats, y = y, device = device, save=False)
        epoch += 1
    
    NMI, ARI = moco_valid(model=MOmoco, dataset_name = args.dataset, data_size=data_size, 
                            class_num = class_num , g_feats = g_feats, y = y, device = device,save=args.save)

if __name__ == "__main__":
    main()
