import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import argparse
import numpy as np
from data import *
from sp_model_k import *
from ChebNet import *
from helper import *
from collections import defaultdict
import time

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--iter', type=int, default=5000, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=10, help='number of samples per setting')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--n_hid', type=int, default=64, help='')
parser.add_argument('--Su', type=int, default=8, help='')
parser.add_argument('--Nr', type=int, default=32, help='Number of receiver')
parser.add_argument('--snr', type=int, default=32, help='snrdb testing')
parser.add_argument('--snrdb_low', type=int, default=25, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=50, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--cheb_order', type=int, default=3, help='order of chebshev')
args = parser.parse_args()

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)

Nt = args.Nr; Nr = args.Nr
ill = True if args.ill == 1 else False
soft_max = nn.Softmax(dim=2)
constellation = generate_QAM_signal_list(args.k)
constellation_expanded = np.expand_dims(constellation, axis=1)
n_class = len(constellation)
cond = 'ill' if ill else 'well'
n_feature = 2
cheb_order = args.cheb_order
EP_iter = 10; GNN_iter = 2
beta = 0.7
ChebNet = ChebNet(EP_iter, beta, args.n_hid, args.Su, constellation, device, dtype)
model = sp_GNN(cheb_order, n_class, GNN_iter, args.n_hid, args.Su, n_feature).to(device)
model_name = '{}x{}_k{}_{}_sp{}'.format(
    Nr, Nr, args.k, cond, cheb_order
)

snr = 10**(args.snr/10.0)
print(model_name)

infer_time_list = []
data_time_list = []
for it in range(args.iter):
    last_time = time.time()
    H, y, x, snr_x, indices, noise_sigma = generate_MIMO_data_batch_ori(
        Nt, Nr, args.k, args.batch_size, snr, snr, ill)
    L, y_hat, alpha = sp_GNN_data(H, y)
    now_time = time.time()
    data_format_time = now_time - last_time
    noise_sigma2 = noise_sigma**2
    all_ones = torch.ones(y.shape)
    H_t_e = torch.matmul(H.permute(0,2,1), all_ones)
    feature = torch.cat((y_hat, alpha * noise_sigma.unsqueeze(-1) * H_t_e), dim=2)
    feature = feature.to(dtype).to(device)
    label = indices.to(device)
    L = L.to(dtype).to(device)
    y = y.squeeze().to(dtype).to(device)
    H = H.to(dtype).to(device)
    noise_sigma2 = noise_sigma2.to(dtype).to(device)
    with torch.no_grad():
        last_time = time.time()
        out = ChebNet.forward(model, H, y, noise_sigma2, feature, L, args.k)
        now_time = time.time()
        inference_time = now_time - last_time
    infer_time_list.append(inference_time)
    data_time_list.append(data_format_time)
    del feature, label, L, y, H, noise_sigma2, out

filename = '../time_compare/spGNNkv2_{}_b{}_iter{}'.format(model_name, args.batch_size, args.iter)
torch.save({'infer':infer_time_list, 'data':data_time_list}, filename)


