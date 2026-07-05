from sklearn.metrics import v_measure_score, adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np
import torch
import pickle
import os


def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    return nmi, ari

def moco_valid(model, dataset_name, data_size, class_num, g_feats, y, device, save):

    model.eval()

    x = [torch.FloatTensor(modality).to(device) for modality in g_feats]

    with torch.no_grad():
        z_a, z_b = model.get_embedding(x[0],x[1])

    z0 = z_a.cpu().detach().numpy()
    z1 = z_b.cpu().detach().numpy()
    
    labels_vector = np.array(y).reshape(data_size)

    y_pred_0 = KMeans(n_clusters=class_num, n_init="auto", random_state=0).fit_predict(z0)
    y_pred_1 = KMeans(n_clusters=class_num, n_init="auto", random_state=0).fit_predict(z1)
    
    nmi, ari = evaluate(labels_vector, y_pred_0)
    print(' View 0 NMI = {:.8f} ARI = {:.8f} '.format(nmi,ari)) 
    nmi, ari = evaluate(labels_vector, y_pred_1)
    print(' View 1 NMI = {:.8f} ARI = {:.8f} '.format(nmi,ari))

    if(save):
        preprocessed_data = {"z0": np.array(z0), "z1": np.array(z1), 
                            'y_pred_0': y_pred_0, 'y_pred_1': y_pred_1}
        pickle.dump(preprocessed_data, open(os.path.join('./lowdimention', f'{dataset_name}_moco_embedding.pkl'), 'wb'))
        print("embedding saved") 

    return nmi, ari

def con_valid(model, adjs_n, g_feats, data_size, class_num, y, save, dataset_name):

    model.eval()

    with torch.no_grad():
        _, _, _, zs = model(adjs_n, g_feats)

    z0 = zs[0].cpu().detach().numpy()
    z1 = zs[1].cpu().detach().numpy()
    z_joint = zs[2].cpu().detach().numpy()
    
    labels_vector = np.array(y).reshape(data_size)

    y_pred_0 = KMeans(n_clusters=class_num, n_init="auto", random_state=0).fit_predict(z0)
    y_pred_1 = KMeans(n_clusters=class_num, n_init="auto", random_state=0).fit_predict(z1)
    y_pred_joint = KMeans(n_clusters=class_num, n_init="auto", random_state=0).fit_predict(z_joint)
    

    nmi, ari = evaluate(labels_vector, y_pred_0)
    print(' View 0 NMI = {:.8f} ARI = {:.8f} '.format(nmi,ari)) 
    nmi, ari = evaluate(labels_vector, y_pred_1)
    print(' View 1 NMI = {:.8f} ARI = {:.8f} '.format(nmi,ari))
    nmi, ari = evaluate(labels_vector, y_pred_joint)
    print(' View Joint NMI = {:.8f} ARI = {:.8f} '.format(nmi,ari))

    if(save):
        preprocessed_data = {"z0": np.array(z0), "z1": np.array(z1), "z_joint": z_joint,
                            'y_pred_0': y_pred_0, 'y_pred_1': y_pred_1, 'y_pred_joint': y_pred_joint}
        pickle.dump(preprocessed_data, open(os.path.join('./lowdimention', f'{dataset_name}_embedding.pkl'), 'wb'))
        print("embedding saved") 

    return nmi, ari

