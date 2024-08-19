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
parser.add_argument('--iter', type=int, default=100, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=1000, help='number of samples per setting')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--n_hid', type=int, default=64, help='')
parser.add_argument('--Su', type=int, default=8, help='')
parser.add_argument('--Nr', type=int, default=16, help='Number of receiver')
parser.add_argument('--snrdb_low', type=int, default=25, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=50, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
parser.add_argument('--epoch', type=int, default=850, help='total number of epoches used in training')
parser.add_argument('--cheb_order', type=int, default=3, help='order of chebshev')
parser.add_argument('--seed', type=int, default=3, help='random seed')
args = parser.parse_args()

def generate_data(batch_size):
    data_dict = {int(NT):{} for NT in NT_list}
    for Nt in NT_list:
        for snrdb in snrdb_list[Nt]: 
            snr = 10**(snrdb/10.0)
            H, y, x, snr, indices, noise_sigma = generate_MIMO_data_batch_ori(
                Nt, Nr, args.k, batch_size, snr, snr, ill)
            L, y_hat, alpha = sp_GNN_data(H, y)
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
            data_dict[int(Nt)][snrdb] = (H, y, L, feature, noise_sigma2, label)
    return data_dict

def eval_model(model, H, y, L, feature, noise_sigma2, label, Nt):
    batch_size = L.shape[0]
    with torch.no_grad():
        last_time = time.time()
        out = ChebNet.forward(model, H, y, noise_sigma2, feature, L, args.k)
        now_time = time.time()
        time_elapsed = now_time - last_time
        y_pred_soft = soft_max(out)
        loss = 0.0
        x_hats = []
        for idx_user in range(label.shape[-1]):
            loss_each = criterion(out[:,idx_user,:], label[:,idx_user]) 
            loss = loss + loss_each
            x_hat = np.matmul(y_pred_soft[:, idx_user, :].to('cpu').detach().numpy(), constellation_expanded)
            x_hats.append(x_hat)
        x_hats = np.concatenate(x_hats,1)
        x_indices = find_index(torch.tensor(x_hats), constellation)
        #acc = (out.argmax(dim=2) == label).cpu().numpy().sum()/(2 * Nt * batch_size)
        acc = (x_indices == label.cpu()).sum()/(x_indices.numel())
    return acc.item(), loss.item(), time_elapsed

torch.manual_seed(args.seed + 12345)

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

BASE_DIR = '../tested_model_lr{}_spk_v2'.format(args.learning_rate)

n_feature = 2
cheb_order = args.cheb_order
EP_iter = 10; GNN_iter = 2
beta = 0.7
ChebNet = ChebNet(EP_iter, beta, args.n_hid, args.Su, constellation, device, dtype)
model = sp_GNN(cheb_order, n_class, GNN_iter, args.n_hid, args.Su, n_feature).to(device)

model_name = '{}x{}_k{}_{}_sp{}'.format(
    Nr, Nr, args.k, cond, cheb_order
)

checkpoint = torch.load('{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_best.pt'.format(
    BASE_DIR, model_name, args.epoch, args.snrdb_low, args.snrdb_high))

model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.CrossEntropyLoss().to(device=device)
NT_list = np.asarray([int(Nt),])
# for each NT, generate data with different snr
snrdb_list = {int(Nt):np.arange(args.snrdb_low, args.snrdb_high + 1)}

result_dict = {int(NT):defaultdict(float) for NT in NT_list}
time_dict = {int(snr):0 for snr in snrdb_list[Nt]}
for it in range(args.iter):
    data_dict = generate_data(args.batch_size)
    for Nt in NT_list:
        for snr in snrdb_list[Nt]:
            print("{} - {} - Nr: {} | Nt: {} | snr: {}".format(it, "spGNN", Nr, Nt, snr))
            # select batch data according to NT and snr
            H, y, L, feature, noise_sigma2, label = data_dict[int(Nt)][snr]
            acc, loss, time_elapsed = eval_model(model, H, y, L, feature, noise_sigma2, label, Nt)
            result_dict[Nt][snr] += acc
            time_dict[snr] += time_elapsed

for NT in NT_list:
    for snr in snrdb_list[NT]:
        result_dict[NT][snr] /= args.iter 

for snr in snrdb_list[NT]:
    time_dict[snr] /= args.iter

filename = '../ml_results_rep/spGNNkv2_{}_ep{}_lr{}_dbmin{}_dbmax{}_seed{}'.format(
    model_name, args.epoch, args.learning_rate, args.snrdb_low, args.snrdb_high, args.seed)
torch.save({'res':result_dict, 'time':time_dict}, filename)
