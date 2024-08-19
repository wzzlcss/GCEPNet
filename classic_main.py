import torch
import numpy as np
import torch.nn as nn
import pickle
from classic_solver import *
from data import *
from collections import defaultdict
from helper import *
import argparse

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--Nr', type=int, default=16, help='Number of receiver')
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--snrdb_low', type=int, default=20, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=60, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--iter', type=int, default=50, help='number of iteration/replication')
parser.add_argument('--batch_size', type=int, default=1000, help='number of samples per setting')
parser.add_argument('--method', type=str, default='ep', help='the name of classic method')
parser.add_argument('--save_result', action='store_true', default=False, help='save result in a file')
parser.add_argument('--seed', type=int, default=3, help='random seed')
args = parser.parse_args()

# generate_MIMO_data_batch(Nt, Nr, k, batch_size, snr_min, snr_max)
def generate_data():
    data_dict = {int(NT):{} for NT in NT_list}
    for NT in NT_list:
        for snrdb in snrdb_list[NT]:
            snr = 10**(snrdb/10.0)
            H_b, y_b, x_b, snr_b, indices_b, noise_sigma_b = generate_MIMO_data_batch_ori(
                int(NT), args.Nr, args.k, batch_size=args.batch_size, snr_min=snr, snr_max=snr, ill=ill)
            data_dict[int(NT)][snrdb] = (H_b, y_b, x_b, noise_sigma_b)
    return data_dict

def test_batch(H_b, y_b, x_b, noise_sigma_b):
    H = H_b.numpy()
    y = y_b.numpy()
    Nt = int(x_b.shape[1]/2)
    constellation = generate_QAM_signal_list(args.k)
    if args.method == "babai":       
        results = babai(H, y, constellation)
        results = torch.tensor(results).unsqueeze(-1)
    if args.method == "ep":
        num_iter = 10
        sigma = noise_sigma_b[0].item()
        sigma2 = sigma**2
        results = EP(y.squeeze(), H, sigma2, num_iter, Nt, constellation, args.k)      
        x_indices = find_index(torch.tensor(results), constellation)
        results = torch.tensor(constellation[x_indices]).unsqueeze(-1)
    if args.method == "oamp":
        num_iter = 10
        sigma = noise_sigma_b[0].item()
        sigma2_c = 2.0 * (sigma**2)
        results = oamp(H, y, sigma2_c, constellation, num_iter)
        x_indices = find_index(torch.tensor(results).squeeze(), constellation)
        results = torch.tensor(constellation[x_indices]).unsqueeze(-1)
    if args.method == "mmse":
        sigma = noise_sigma_b[0].item()
        sigma2 = sigma**2
        results = mmse(y_b.squeeze(), H_b, sigma2, args.k)
        x_indices = find_index(results, constellation)
        results = torch.tensor(constellation[x_indices]).unsqueeze(-1)
    accr = accuracy(results, x_b)
    return accr

torch.manual_seed(12345 + args.seed)
NT_list = np.asarray([int(args.Nr),])
# for each NT, generate data with different snr
snrdb_list = {int(args.Nr):np.arange(args.snrdb_low, args.snrdb_high + 1)} # 10 log_10 (snr)
ill = True if args.ill == 1 else False
NT = int(args.Nr)
name = 'ill' if ill else 'well'
filename = './classic_results_rep/{}_Nr{}_Nt{}_k{}_{}_{}.pt'.format(
    args.method, args.Nr, NT, args.k, name, args.seed)

result_dict = {int(NT):defaultdict(float) for NT in NT_list}
for it in range(args.iter):
    data_dict = generate_data()
    for NT in NT_list:
        for snrdb in snrdb_list[NT]:
            print("{} - {} - Nr: {} | Nt: {} | snr (db): {}".format(
                it, args.method, args.Nr, NT, snrdb))
            # select batch data according to NT and snr
            H_b, y_b, x_b, noise_sigma_b = data_dict[int(NT)][snrdb]
            accr = test_batch(H_b, y_b, x_b, noise_sigma_b)
            result_dict[NT][snrdb] += accr

for NT in NT_list:
    for snrdb in snrdb_list[NT]:
        result_dict[NT][snrdb] /= args.iter

torch.save(result_dict, filename)
