# import os, sys
# sys.path.append('../')
from data import *
import numpy as np
import torch

def generate_MIMO_data_batch_GEPNet(Nt, Nr, k, batch_size, snr_min, snr_max, ill):
    H, y, x, snr, indices, noise_sigma = generate_MIMO_data_batch_ori(
        Nt, Nr, k, batch_size, snr_min, snr_max, ill)
    noise_sigma2 = noise_sigma**2
    init_feats, edge_i_j_feats = Feature_gens(
        np.array(y.squeeze()), np.array(H), np.array(noise_sigma2), Nt, batch_size)
    edge_i_j_feats = edge_i_j_feats.reshape(batch_size, -1)
    # to tensor
    init_feats = torch.from_numpy(init_feats)
    edge_weight = torch.from_numpy(edge_i_j_feats)
    return H, y.squeeze(), indices, init_feats, edge_weight, noise_sigma2

def Feature_gens(y, H, noiseLevel, Nt, batch_size):
    # init_feats = np.ones((self.batch_size,self.Nt*2,3))
    edge_i_j_feats = np.ones((batch_size, Nt*2, (Nt*2-1)))
    yTh = np.matmul(np.expand_dims(y, 2).transpose(0, 2, 1), H) 
    hTh = np.matmul(H.transpose(0, 2, 1), H)
    diag_hTh = np.expand_dims(hTh.diagonal(0, 1, 2), 2).transpose(0, 2, 1)
    noise_arr = np.tile(np.expand_dims(noiseLevel, 2),[1, 1, Nt*2])
    init_feats = np.concatenate((yTh, 1*diag_hTh, noise_arr), 1)
    init_feats = init_feats.transpose(0, 2, 1)   
    for u_idx  in range(Nt*2):
        t=0
        for j_idx  in range(Nt*2):
            if np.not_equal(j_idx, u_idx):
                edge_i_j_feat = np.matmul(
                    np.expand_dims(H[:, :, j_idx], 2).transpose(0, 2, 1),
                    np.expand_dims(H[:, :, u_idx], 2))
                edge_i_j_feats[:, u_idx, t] = np.squeeze(edge_i_j_feat)
                t = t + 1;        
    return init_feats, edge_i_j_feats

def get_idx_dicts(user_num):
    #slicing parameters
    temp_a=[]
    temp_b=[]
    dict_index_val = {}
    dict_val_index = {}
    dict_final = {}  
    for i in range(user_num):
        for j in range(user_num):
            if  i!=j:
                temp_a.append(i)
                temp_b.append(j)
                dict_final[len(temp_a)-1]=  str(j) + str(i)
                dict_index_val[len(temp_a)-1] = str(i) + str(j)
                dict_val_index[str(i) + str(j)] = len(temp_a)-1           
    return temp_a,temp_b
