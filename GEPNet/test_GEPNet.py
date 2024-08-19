import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import argparse
import numpy as np
from data import *
from GEPNet_data import *
from GEPNet import GEPNet
from GEPNet_GNN import GNN
from helper import *
from collections import defaultdict
import time

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--iter', type=int, default=100, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=1000, help='number of samples per setting')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--num_neuron', type=int, default=64, help='num_neuron')
parser.add_argument('--su','-su', type=int, default=8, help='num_feature_su')
parser.add_argument('--beta', type=float, default=0.7, help='beta_EP')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
parser.add_argument('--Dropout', type=float, default=0, help='dropout')
parser.add_argument('--iter_GEPNet', type=int, default=10, help='number of GEPNet iterations ')
parser.add_argument('--iter_GNN', type=int, default=2, help='number of GNN iterations inside a GEPNet iteration')
parser.add_argument('--Nr', type=int, default=16, help='Number of receiver')
parser.add_argument('--snrdb_low', type=int, default=25, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=50, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--epoch', type=int, default=850, help='total number of epoches used in training')
parser.add_argument('--seed', type=int, default=3, help='random seed')
args = parser.parse_args()

def generate_data(batch_size):
    data_dict = {int(NT):{} for NT in NT_list}
    for Nt in NT_list:
        for snrdb in snrdb_list[Nt]: 
            snr = 10**(snrdb/10.0)
            H, y, x_label, init_feats, edge_weight, noise_sigma2 = generate_MIMO_data_batch_GEPNet(
                Nt, Nr, args.k, batch_size, snr, snr, ill)            
            data_dict[int(Nt)][snrdb] = (H, y, x_label, init_feats, edge_weight, noise_sigma2)
    return data_dict

def eval_model(model, H, y, x_label, init_feats, edge_weight, noise_sigma2, Nt):
    with torch.no_grad():
        init_feats=init_feats.to(dtype).to(device)
        edge_weight=edge_weight.to(dtype).to(device)
        x_label=x_label.to(device)
        noise_sigma2 = noise_sigma2.to(dtype).to(device)
        H = H.to(dtype).to(device)
        y = y.to(dtype).to(device)       
        x_hats = []
        last_time = time.time()
        y_pred = GEPNet.forward(model, H, y, noise_sigma2, init_feats, edge_weight, args.k)
        now_time = time.time()
        time_elapsed = now_time - last_time
        y_pred_soft = soft_max(y_pred)
        loss = 0.0
        for idx_user in range(x_label.shape[-1]):            
            loss_each = criterion(y_pred[:, idx_user, :], x_label[:, idx_user]) 
            loss = loss + loss_each
            constellation_expanded = np.expand_dims(constellation, axis=1)
            x_hat = np.matmul(y_pred_soft[:, idx_user, :].to('cpu').detach().numpy(), constellation_expanded)
            x_hats.append(x_hat)
        x_hats = np.concatenate(x_hats,1)
        x_indices = find_index(torch.tensor(x_hats), constellation)
        accr = (x_indices == x_label.cpu()).sum()/(x_indices.numel())
        return accr.item(), loss.item(), time_elapsed

torch.manual_seed(args.seed + 12345)

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

BASE_DIR = '../tested_model_lr{}'.format(args.learning_rate)

checkpoint = torch.load('{}/GEPNet_{}_ep{}_dbmin{}_dbmax{}_best.pt'.format(
    BASE_DIR, model_name, args.epoch, args.snrdb_low, args.snrdb_high))

GEPNet = GEPNet(args.iter_GEPNet, args.beta, Nr, args.su, args.num_neuron, constellation, device, dtype)
model = GNN(args.iter_GNN, args.num_neuron, args.su, num_classes, args.Dropout).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.CrossEntropyLoss().to(device=device)
NT_list = np.asarray([int(Nt),])

# for each NT, generate data with different snr
snrdb_list = {int(Nt):np.arange(args.snrdb_low, args.snrdb_high + 1)}

result_dict = {int(NT):defaultdict(float) for NT in NT_list}
time_dict = {int(snr):0 for snr in snrdb_list[Nt]}
for it in range(args.iter):
    data_dict = generate_data(args.batch_size)
    for NT in NT_list:
        for snr in snrdb_list[NT]:
            print("{} - {} - Nr: {} | Nt: {} | snr: {}".format(it, "GEPNet", Nr, NT, snr))
            # select batch data according to NT and snr
            H, y, x_label, init_feats, edge_weight, noise_sigma2 = data_dict[int(NT)][snr]
            acc, loss, time_elapsed = eval_model(model, H, y, x_label, init_feats, edge_weight, noise_sigma2, Nt)
            result_dict[NT][snr] += acc
            time_dict[snr] += time_elapsed

for NT in NT_list:
    for snr in snrdb_list[NT]:
        result_dict[NT][snr] /= args.iter 

for snr in snrdb_list[NT]:
    time_dict[snr] /= args.iter

filename = '../ml_results_rep/GEPNet_{}_ep{}_lr{}_dbmin{}_dbmax{}_seed{}'.format(
    model_name, args.epoch, args.learning_rate, args.snrdb_low, args.snrdb_high, args.seed)
torch.save({'res':result_dict, 'time':time_dict}, filename)
