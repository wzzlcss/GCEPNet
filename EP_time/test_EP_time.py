import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import argparse
import numpy as np
from data import *
from helper import *
from EP_cuda import *
from collections import defaultdict
import time

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--iter', type=int, default=5000, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=10, help='number of samples per setting')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--beta', type=float, default=0.7, help='beta_EP')
parser.add_argument('--iter_EP', type=int, default=10, help='number of EP iterations ')
parser.add_argument('--Nr', type=int, default=64, help='Number of receiver')
parser.add_argument('--snr', type=int, default=32, help='snrdb testing')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
args = parser.parse_args()

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)

Nt = args.Nr; Nr = args.Nr
ill = True if args.ill == 1 else False
soft_max = nn.Softmax(dim=2)
constellation = generate_QAM_signal_list(args.k)
num_classes = len(constellation)
cond = 'ill' if ill else 'well'
model_name = '{}x{}_k{}_{}'.format(
    Nr, Nr, args.k, cond
)

EPNet = EPNet(args.iter_EP, args.beta, constellation, device, dtype)

snr = 10**(args.snr/10.0)
print(model_name)

infer_time_list = []
data_time_list = []
for it in range(args.iter):
    last_time = time.time()
    # H, y, x_label, init_feats, edge_weight, noise_sigma2 = generate_MIMO_data_batch_GEPNet(
    #     Nt, Nr, args.k, args.batch_size, snr, snr, ill) 
    H, y, x, snr_x, indices, noise_sigma = generate_MIMO_data_batch_ori(
        Nt, Nr, args.k, args.batch_size, snr, snr, ill)
    noise_sigma2 = noise_sigma**2
    now_time = time.time()
    data_format_time = now_time - last_time
    noise_sigma2 = noise_sigma2.to(dtype).to(device)
    H = H.to(dtype).to(device)
    y = y.squeeze(-1).to(dtype).to(device)       
    with torch.no_grad():
        last_time = time.time()
        out = EPNet.forward(H, y, noise_sigma2, args.k)
        now_time = time.time()
        inference_time = now_time - last_time
    infer_time_list.append(inference_time)
    data_time_list.append(data_format_time)

filename = '../time_compare/EPcuda_{}_b{}_iter{}'.format(model_name, args.batch_size, args.iter)
torch.save({'infer':infer_time_list, 'data':data_time_list}, filename)
