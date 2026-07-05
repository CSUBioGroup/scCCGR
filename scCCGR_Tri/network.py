import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features

        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        self.fc = nn.Linear(latent_features[-1], num_cluster)
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return out
    
    def get_embedding(self, x):
        latent = self.encoder(x)

        return latent

class MoCo(nn.Module):
    def __init__(self, 
                 encoder,
                 in_features, 
                 num_cluster,
                 latent_features=[1024, 512, 128],
                 mlp=True,
                 K=65536, 
                 m=0.999, 
                 T=0.9,  
                 p=0.0, 
                 lam=0.1, 
                 alpha=0.1): 
        super().__init__()
        self.K = int(K)
        self.m = m
        self.T = T
        self.lam = lam
        self.alpha = alpha
        self.rep_dim = latent_features[-1]
        
        self.encoder_q = encoder(in_features=in_features,
                                 num_cluster=num_cluster, 
                                 latent_features=latent_features,
                                 p=p)
        self.encoder_k = encoder(in_features=in_features, 
                                 num_cluster=num_cluster,
                                 latent_features=latent_features,
                                 p=p)
        
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )

            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )

        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", 
                             F.normalize(torch.randn(self.K, self.rep_dim, requires_grad=False), dim=1))
        self.ptr = 0
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        
        self.queue[self.ptr: self.ptr + batch_size, :] = keys.detach()
        self.ptr = (self.ptr + batch_size) % self.K
        self.queue.requires_grad = False

    def forward_aug_nn(self, x1, x2): 
        q = self.encoder_q(x1)
        latent = self.encoder_q.get_embedding(x1)
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)
        qc = q.unsqueeze(1) 
        for _ in range(1, c):  
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)
        qc = qc.reshape(-1, q.size(1)) 

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)

            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        pos_sim1 = (1 - self.lam) * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1) # [B, 1]
        pos_sim2 = (self.lam / c) * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1) # [B*c, 1]
        pos_sim2 = pos_sim2.reshape(-1, c) 

        assert pos_sim2.size(0) == pos_sim1.size(0)

        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        loss = -(torch.logsumexp(pos_sim / self.T , dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        penalty = self.alpha * (torch.mean(torch.abs(latent))) 
        loss += penalty 

        self._dequeue_and_enqueue(k2)

        return loss
    
    def forward(self, x1, x2, flag="aug_nn"):
        if flag == 'aug_nn':
            return self.forward_aug_nn(x1, x2)

        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1)

        pos_sim = torch.einsum("ic, ic -> i", [q, k]).unsqueeze(-1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        self._dequeue_and_enqueue(k)

        return logits, labels
     
    def get_embedding(self, x):
        out = self.encoder_q.get_embedding(x)
        
        return out

class MultiOmicsMoCo(nn.Module):
    def __init__(self,
                 encoder,
                 in_features1,
                 in_features2,
                 in_features3,
                 num_cluster,
                 latent_features1=[128, 64, 32],
                 latent_features2=[128, 64, 32],
                 latent_features3=[128, 64, 32],
                 K=65536,
                 m=0.999,
                 T=0.9,
                 p=0.0,
                 lam=0.1,
                 alpha=0.1
                 ):   
        
        super().__init__()

        self.moco1 = MoCo(
            encoder=encoder,
            in_features=in_features1,
            num_cluster=num_cluster,
            latent_features=latent_features1,
            K=K,
            m=m,
            T=T,
            p=p,
            lam=lam,
            alpha=alpha
        )

        self.moco2 = MoCo(
            encoder=encoder,
            in_features=in_features2,
            num_cluster=num_cluster,
            latent_features=latent_features2,
            K=K,
            m=m,
            T=T,
            p=p,
            lam=lam,
            alpha=alpha
        )
        self.moco3 = MoCo(
            encoder=encoder,
            in_features=in_features3,
            num_cluster=num_cluster,
            latent_features=latent_features3,
            K=K,
            m=m,
            T=T,
            p=p,
            lam=lam,
            alpha=alpha
        )

    def forward(self, x1, x2, x3, x1_aug, x2_aug, x3_aug, flag="aug_nn"):

        loss1 = self.moco1(x1, x1_aug, flag=flag)
        loss2 = self.moco2(x2, x2_aug, flag=flag)
        loss3 = self.moco3(x3, x3_aug, flag=flag)

        loss = loss1 + loss2 + loss3

        return loss, loss1, loss2, loss3

    @torch.no_grad()
    def get_embedding(self, x1, x2, x3):

        z1 = self.moco1.get_embedding(x1)
        z2 = self.moco2.get_embedding(x2)
        z3 = self.moco3.get_embedding(x3)

        return z1, z2, z3   

class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()

        self.in_feat = in_feat

        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))

        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2, emb3):

        emb = []

        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb3), dim=1))

        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))

        self.vu = torch.matmul(self.v, self.u_omega)

        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1)

        emb_combined = torch.matmul(
            torch.transpose(self.emb, 1, 2),
            torch.unsqueeze(self.alpha, -1)
        )

        return torch.squeeze(emb_combined), self.alpha


class GCNConv(nn.Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNModelAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim, dropout, act=F.relu)
        self.gc2 = GCNConv(hidden_dim, latent_dim, dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, encode=False):
        z = self.encode(x, adj)
        return z
    
class adjDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(adjDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Network(nn.Module):
    def __init__(self, 
                 view, 
                 input_dims,
                 mid_layer_dims,
                 feature_dim, 
                 high_feature_dim, 
                 class_num, 
                 dropout):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.dropout = dropout
        self.view = view
        self.atten_feat = input_dims[view-1]

        self.atten_cross = AttentionLayer(self.atten_feat, self.atten_feat)

        for v in range(view):
            self.encoders.append(GCNModelAE(input_dims[v], 
                                            mid_layer_dims[v],
                                            feature_dim,
                                            self.dropout))

            self.decoders.append(adjDecoder(self.dropout, act=lambda x: x))  # lambda x: x torch.sigmoid
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
            
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, adjs_n, g_feats):
        x0 = g_feats[0]
        x1 = g_feats[1]
        x2 = g_feats[2]

        x_joint,atten_score = self.atten_cross(x0, x1, x2)
        feats = [x0, x1, x2, x_joint]

        hs = []
        qs = []
        xrs = []
        zs = []

        for v in range(self.view):
            adj, feat = adjs_n[v], feats[v]
            z = self.encoders[v](feat, adj) 
            h = normalize(self.feature_contrastive_module(z), dim=1) 
            q = self.label_contrastive_module(z) 
            xr = self.decoders[v](z) 
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

