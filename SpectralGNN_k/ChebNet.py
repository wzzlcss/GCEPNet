import torch
from EP import EP
from sp_model_k import *

class ChebNet(object):
    def __init__(self, iter_GEP, beta, num_neuron, su, cons, device, dtype):  

        self.iter_GEP = iter_GEP
        self.beta = beta
        self.hid_neuron_size= num_neuron
        self.su = su
        self.cons = torch.from_numpy(cons).to(dtype).to(device) # constellation set
        self.device = device
        self.dtype = dtype

    def forward(self, model, H, y, sigma2, signal, L, k):
        # u_feats: initial features
        # edge: edge weights

        # Initialization
        user_num = H.shape[2] # num of x
        batch_size = H.shape[0]
        # (batch_size, x_num, 1)
        lamda = (torch.ones((batch_size, user_num, 1))*(3.0/(4**k - 1))).to(self.dtype).to(self.device)
        gamma = torch.zeros((batch_size, user_num, 1)).to(self.dtype).to(self.device)

        mean_ab = None
        var_ab = None
        p_GNN = None
        read_gru = None
        gru = torch.zeros((batch_size, user_num, self.hid_neuron_size)).to(self.dtype).to(self.device)
        EP_model = EP(H, y, sigma2, user_num, self.cons, batch_size)

        # GEPNet iteration
        for iteration in range(self.iter_GEP):
            # (batch_size, x_num, x_num)
            diag_lamda = torch.zeros((batch_size, user_num, user_num)).to(self.dtype).to(self.device)
            #Perform EP: mean_ab, var_ab are from the cavity marginal
            mean_ab, var_ab, lamda, gamma = EP_model.performEP(self.beta, diag_lamda, p_GNN, mean_ab, var_ab, lamda, gamma, iteration)
            #Perform GNN (model: based on cheby)
            p_GNN, read_gru, gru = model(read_gru, gru, signal, L, sigma2, mean_ab, var_ab, iteration)

        return p_GNN