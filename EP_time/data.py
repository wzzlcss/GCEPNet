import numpy as np
import scipy
import math
import torch
import torch.utils.data
import random
from scipy.linalg import sqrtm
# import sys
# sys.path.append("..")
# from classic_solver import *
# from helper import *
# from data import generate_MIMO_data

# only consider Nt <= Nr

# generate noise following Gaussian distribution
def generate_noise(Nr, noise_sigma): # noise_sigma is the std for the complex noise
    v = np.random.normal(0, noise_sigma, 2*Nr)
    return v

# generate random channel and each element is sampled from i.i.d gaussian distribution
def generate_gaussian_channel(Nt, Nr, mu=0):
    sigma = 1.0/np.sqrt(2. * Nr)
    Hr =  np.random.normal(mu, sigma, size=(Nr, Nt))
    Hi =  np.random.normal(mu, sigma, size=(Nr, Nt))
    h1 = np.concatenate((Hr, -1.0 * Hi), axis = 1)
    h2 = np.concatenate((Hi, Hr), axis = 1)
    H = np.concatenate((h1, h2), axis = 0)   
    return H

def ill_helper(N, x):
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            matrix[i, j] = x ** (abs(i - j))
    return matrix

def generate_ill_channel(Nt, Nr, a=0.95, b=0.95):
    H = np.random.normal(0, 1.0/np.sqrt(2.0 * Nr), size=(2*Nr, 2*Nt))
    psi = ill_helper(2*Nr, a)
    phi = ill_helper(2*Nt, b)
    psi_2 = sqrtm(psi)
    phi_2 = sqrtm(phi)
    A = psi_2 @ H @ phi_2
    return A

# generate the valid signal values w.r.t different QAM constellation
# k = 1, 2, 3 correspond to QPSK (i.e. 4QAM), 16QAM, 64QAM
def generate_QAM_signal_list(k): # unnormalized
    vaild_signal_list = np.linspace(int(-(2**k -1)), int(2**k - 1), int(2**k))
    return vaild_signal_list

# generate x by randomly selecting signal values from valid signal list
def generate_rand_index(constellation, Nt):
    indice = [np.random.choice(range(len(constellation))) for i in range(2 * Nt)]
    # s = constellation[indice]
    return indice

# Nt = 3; Nr = 4; k = 2; batch_size = 5; snr_min = 1; snr_max = 5
# generate a batch of samples, Nr <= Nr
def generate_MIMO_data_batch_ori(Nt, Nr, k, batch_size, snr_min, snr_max, ill=False):
    if ill:
        channel_matrix = np.array([generate_ill_channel(Nt, Nr) for i in range(batch_size)])
    else:
        channel_matrix = np.array([generate_gaussian_channel(Nt, Nr, mu=0) for i in range(batch_size)])
    snr = torch.empty((batch_size, 1)).uniform_(snr_min, snr_max)
    channel_sigma2 = 1.0 / Nr
    noise_sigma2 = 2.0 * Nt * (4**k - 1) * channel_sigma2 / (3.0 * snr) 
    noise_sigma2 = noise_sigma2 * 0.5
    noise_sigma = np.sqrt(noise_sigma2)
    noise = np.array([generate_noise(Nr, noise_sigma_i) for noise_sigma_i in noise_sigma])
    constellation = generate_QAM_signal_list(k)
    indices = np.array([generate_rand_index(constellation, Nt) for i in range(batch_size)])
    x = constellation[indices]
    y = np.array([channel_matrix[i] @ x[i] + noise[i] for i in range(batch_size)])
    x = torch.from_numpy(x).to(dtype=torch.float64)
    indices = torch.from_numpy(indices).to(dtype=torch.int64)
    y = torch.from_numpy(y).to(dtype=torch.float64)
    y = y.unsqueeze(-1); x = x.unsqueeze(-1)
    channel_matrix = torch.from_numpy(channel_matrix).to(dtype=torch.float64)
    return channel_matrix, y, x, snr, indices, noise_sigma

# transform the data for the spectral method
def sp_GNN_data(channel_matrix, y):
    H = channel_matrix
    two_NT = int(H.shape[-1])
    batch_size = int(H.shape[0])
    H_t = H.permute(0,2,1)
    HTH = torch.matmul(H_t, H)
    lambda_max = [max(torch.linalg.eigh(HTH[i])[0]).item() for i in range(batch_size)]
    lambda_max = torch.tensor(np.array(lambda_max))
    alpha = 1.0/lambda_max
    # compute L = I - alpha HTH, y = alpha HT y
    alpha = alpha.view(-1, 1, 1)
    y_hat = alpha * torch.matmul(H_t, y)
    scaled_HTH = alpha * HTH
    I = torch.eye(two_NT).unsqueeze(0).repeat(batch_size, 1, 1)
    L = I - scaled_HTH
    return L, y_hat


# A = HTH[0]
# size = A.shape[0]
# L = (torch.eye(size) - A).numpy()
# eigenvalues, eigenvectors = np.linalg.eig(L)
# scale_L = torch.eye(size) - alpha[0] * A
# eigenvalues2, eigenvectors2 = np.linalg.eig(scale_L)

#D = torch.sum(HTH, dim=-1)
#inv_D = torch.pow(D, -1.0)
#mtx_D = torch.diag_embed(inv_D)
#norm_HTH = torch.matmul(mtx_D, HTH) # D^-1 H^T H
#inv_D_HT = torch.matmul(mtx_D, H_t)
#y_hat = torch.matmul(inv_D_HT, y) # D^-1 H^T y

def find_qpsk_single(x, k, Nt):
    QAM_16 = {3:[1, 1], 1:[-1, 1], -3:[-1, -1], -1:[1, -1]}
    QAM_64 = {7:[1, 1, 1], 5:[-1, 1, 1], 3:[1, -1, 1], 1:[-1, -1, 1], -7:[-1, -1, -1], -5:[1, -1, -1], -3:[-1, 1, -1], -1:[1, 1, -1]}
    if k == 2: # QAM_16 to QPSK
        res = [QAM_16[xi] for xi in x]
    elif k == 3: # QAM_64 to QPSK
        res = [QAM_64[xi] for xi in x]
    res = np.array(res).T.reshape(Nt * k)
    return res

def find_qpsk_batch(x_b, k):
    Nt = x_b.shape[1]
    x_qpsk = np.array([find_qpsk_single(x, k, Nt) for x in x_b])
    return x_qpsk # array

def qpsk_to_ori_batch(x_qpsk, k, batch_size, NT):
    two_NT = 2 * NT
    # x_qpsk: the output of find_qpsk_batch with shape (batch_size, k*two_NT)
    coef = np.expand_dims(np.array([2**i for i in range(k)]), axis = 0) # (1, k)
    coef = np.expand_dims(np.repeat(coef, batch_size, axis = 0), axis = 1)
    x_ori = np.squeeze(coef @ x_qpsk.reshape(batch_size, k, two_NT), axis = 1) # (batch_size, two_NT)
    return x_ori

def to_qpsk_prob_batch(H_b, y_b, k, batch_size, noise_sigma_b, NT):
    two_NT = 2 * NT
    H_0 = H_b
    H_new = H_0
    for i in range(1, k):
        H_i = (2**i) * H_0
        H_new = torch.cat((H_new, H_i), axis = 2)
    # add constant regularization
    alpha = noise_sigma_b.view(-1, 1, 1)
    I = torch.eye(two_NT * k).unsqueeze(0).repeat(batch_size, 1, 1)
    H_trans = torch.cat((H_new, alpha * I), axis = 1)
    y_trans = torch.cat((y_b, torch.zeros(batch_size, two_NT * k, 1)), axis = 1)
    return H_trans, y_trans

# check correctness of transformation
# x_trans = find_qpsk_batch(x_b.squeeze().numpy(), k)
# # H_new.numpy() @ np.expand_dims(x_trans, axis=2) - H_b.numpy() @ x_b.numpy() should be equal
# coef = np.expand_dims(np.array([2**i for i in range(k)]), axis = 0) # (1, k)
# coef = np.expand_dims(np.repeat(coef, batch_size, axis = 0), axis = 1)
# np.squeeze(coef @ x_trans.reshape(batch_size, k, two_NT), axis = 1) - x_b.squeeze(-1).numpy()


