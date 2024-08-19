import sys
sys.path.append('..')
from data import *
from sp_model_k import *
from ChebNet import *
import numpy as np
import torch
import torch.nn as nn
import argparse
from helper import *

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='task')
# MIMO data parameters
parser.add_argument('--load_trained', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--k', type=int, default=3, help='QAM constellation, k=1 for 4QAM, 2 for 16QAM, 3 for 64QAM')
parser.add_argument('--epoch', type=int, default=850, help='total number of epoches')
parser.add_argument('--epoch_iteration', type=int, default=100, help='number of batches in an epoch')
parser.add_argument('--train_batch_size', type=int, default=100, help='number of samples in a training batch')
parser.add_argument('--valid_batch_size', type=int, default=2000, help='number of samples in a validation batch')
parser.add_argument('--starting_iteration', type=int, default=1, help='use it for continued training')
parser.add_argument('--step', type=int, default=10000, help='number of batches in an epoch')
parser.add_argument('--epoch_valid', type=int, default=100, help='number of batches in an epoch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning_rate')
parser.add_argument('--n_hid', type=int, default=64, help='')
parser.add_argument('--Su', type=int, default=8, help='')
parser.add_argument('--snrdb_low', type=int, default=25, help='lower bound of snrdb training')
parser.add_argument('--snrdb_high', type=int, default=50, help='lower bound of snrdb training')
parser.add_argument('--ill', type=int, default=0, help='1 for ill-conditioned; 0 for random')
parser.add_argument('--Nr', type=int, default=16, help='Number of receiver')
parser.add_argument('--resume_training', type=int, default=0, help='1 for resume training from last saved model')
parser.add_argument('--cheb_order', type=int, default=3, help='order of chebshev')
args = parser.parse_args()

# create folder to store results
BASE_DIR = '../tested_model_lr{}_spk_v2'.format(args.learning_rate)

def train(model, optimizer, device, best_valid_acc, lr_scheduler):
    model.train()
    if args.starting_iteration == 1:
        print('starting tracking validation set')
        # initial save
        torch.save({
            'valid_acc_track': [],
            'valid_loss_track': [],
        }, '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_valid_tracker.pt'.format(
            BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))
    print('starting training from iteration {}'.format(args.starting_iteration))
    # training
    for i in range(args.starting_iteration, total_iter+1):
        Nt = np.random.choice(Nt_list, p=Nt_prob)
        snrdb_min, snrdb_max = args.snrdb_low, args.snrdb_high
        snr_min = 10**(snrdb_min/10.0); snr_max = 10**(snrdb_max/10.0)
        # generate sample
        H, y, x, snr, indices, noise_sigma = generate_MIMO_data_batch_ori(
            Nt, Nr, args.k, args.train_batch_size, snr_min, snr_max, ill)
        L, y_hat, alpha = sp_GNN_data(H, y)
        noise_sigma2 = noise_sigma**2
        # (batch_size, 2Nt, 2)
        all_ones = torch.ones(y.shape)
        H_t_e = torch.matmul(H.permute(0,2,1), all_ones)
        # noise_sigma.unsqueeze(-1) * H_t_e
        feature = torch.cat((y_hat, alpha * noise_sigma.unsqueeze(-1) * H_t_e), dim=2)
        feature = feature.to(dtype).to(device)
        label = indices.to(device)
        L = L.to(dtype).to(device)
        y = y.squeeze().to(dtype).to(device)
        H = H.to(dtype).to(device)
        noise_sigma2 = noise_sigma2.to(dtype).to(device)
        out = ChebNet.forward(model, H, y, noise_sigma2, feature, L, args.k)
        y_pred_soft = soft_max(out)
        loss = 0.0
        x_hats = []
        for idx_user in range(label.shape[-1]):
            loss_each = criterion(out[:,idx_user,:], label[:,idx_user]) 
            x_hat = np.matmul(y_pred_soft[:, idx_user, :].to('cpu').detach().numpy(), constellation_expanded)
            loss = loss + loss_each
            x_hats.append(x_hat)
        x_hats = np.concatenate(x_hats,1)
        x_indices = find_index(torch.tensor(x_hats), constellation)
        acc = (x_indices == label.cpu()).sum()/(x_indices.numel())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iter: {} train loss: {} train acc: {}'.format(i, loss.item(), acc))
        if (i%args.epoch_valid==0):
            # validation
            model.eval()
            valid_loss, valid_acc = validation(model, valid_set, valid_NT_list, valid_snrdb_list)
            # track valid_acc, valid_loss
            checkpoint = torch.load(
                '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_valid_tracker.pt'.format(
                    BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))
            valid_acc_track = checkpoint['valid_acc_track']
            valid_loss_track = checkpoint['valid_loss_track']
            valid_acc_track.append(valid_acc)
            valid_loss_track.append(valid_loss)
            torch.save({
                'valid_acc_track': valid_acc_track,
                'valid_loss_track': valid_loss_track,
            }, '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_valid_tracker.pt'.format(
                BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))
            print('--------------------------iter: {} valid loass: {} valid acc: {}'.format(
                i, valid_loss, valid_acc))
            # adjust learning rate
            if (i%args.step==0):
                lr_scheduler.step(valid_loss)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                #save the best model
                torch.save({
                    'args': args,
                    'iter': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_valid_acc': best_valid_acc
                }, '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}_best.pt'.format(
                    BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))
            model.train()
        torch.save({
            'args': args,
            'iter': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_acc': best_valid_acc
        }, '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}.pt'.format(
            BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))

# generate_MIMO_data_batch(Nt, Nr, k, batch_size, snr_min, snr_max)
def generate_valid_data(batch_size, Nr, k, ill, valid_NT_list, valid_snrdb_list):
    data_dict = {int(NT):{} for NT in valid_NT_list}
    for Nt in valid_NT_list:
        for snrdb in valid_snrdb_list[Nt]:
            snr = 10**(snrdb/10.0)
            H, y, x, snr, indices, noise_sigma = generate_MIMO_data_batch_ori(
                Nt, Nr, k, batch_size, snr, snr, ill)
            L, y_hat, alpha = sp_GNN_data(H, y)
            noise_sigma2 = noise_sigma**2
            # (batch_size, 2Nt, 2)
            all_ones = torch.ones(y.shape)
            H_t_e = torch.matmul(H.permute(0,2,1), all_ones)
            # noise_sigma.unsqueeze(-1) * H_t_e
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
        out = ChebNet.forward(model, H, y, noise_sigma2, feature, L, args.k)
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
        acc = (x_indices == label.cpu()).sum()/(x_indices.numel())
    return acc, loss.item()

def validation(model, valid_set, valid_NT_list, valid_snrdb_list):
    acc_list = []
    loss_list = []
    count = 0
    for Nt in valid_NT_list:
        for snrdb in valid_snrdb_list[Nt]:
            H, y, L, feature, noise_sigma2, label = valid_set[int(Nt)][snrdb]
            acc, loss = eval_model(model, H, y, L, feature, noise_sigma2, label, Nt)
            acc_list.append(acc)
            loss_list.append(loss)
            count += 1
    return np.sum(loss_list)/count, np.sum(acc_list)/count

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)

Nt_list = np.array([int(args.Nr),])
Nt_prob = Nt_list/Nt_list.sum()
Nr = int(args.Nr) # same for validation and for training
total_iter = args.epoch * args.epoch_iteration
ill = True if args.ill == 1 else False

constellation = generate_QAM_signal_list(args.k)
constellation_expanded = np.expand_dims(constellation, axis=1)
n_class = len(constellation)
criterion = nn.CrossEntropyLoss().to(device=device)
soft_max = nn.Softmax(dim=2)

n_feature = 2
cheb_order = args.cheb_order
EP_iter = 10; GNN_iter = 2
beta = 0.7
ChebNet = ChebNet(EP_iter, beta, args.n_hid, args.Su, constellation, device, dtype)
model = sp_GNN(cheb_order, n_class, GNN_iter, args.n_hid, args.Su, n_feature).to(device)

# model name
cond = 'ill' if ill else 'well'
model_name = '{}x{}_k{}_{}_sp{}'.format(
    Nr, Nr, args.k, cond, cheb_order
)
args.model_name = model_name

# generate validation set
valid_NT_list = np.asarray([int(args.Nr), ])
valid_snrdb_list = {int(args.Nr):np.arange(args.snrdb_low, args.snrdb_high + 1)}
# valid_set = generate_valid_data(args.valid_batch_size, Nr, args.k, ill, valid_NT_list, valid_snrdb_list)
# torch.save(valid_set, 
#     '../valid_set/spGNNv2_format_{}x{}_k{}_{}_dbmin{}_dbmax{}'.format(
#         Nr, Nr, args.k, cond, args.snrdb_low, args.snrdb_high
# ))
valid_set = torch.load(
    '../valid_set/spGNNv2_format_{}x{}_k{}_{}_dbmin{}_dbmax{}'.format(
        Nr, Nr, args.k, cond, args.snrdb_low, args.snrdb_high
))

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08)

best_valid_acc = 0.0

if args.resume_training == 1:
    checkpoint = torch.load(
        '{}/spGNNkv2_{}_ep{}_dbmin{}_dbmax{}.pt'.format(
            BASE_DIR, args.model_name, args.epoch, args.snrdb_low, args.snrdb_high))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_valid_acc = checkpoint['best_valid_acc']
    args.starting_iteration = checkpoint['iter'] + 1
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,'min', 0.91, 0, 0.0001, 'rel', 0, 0, 1e-08)

print(args)
train(model, optimizer, device, best_valid_acc, lr_scheduler)