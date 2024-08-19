import torch
import torch.nn as nn
dtype = torch.float64
torch.set_default_dtype(dtype)
from torch.nn import Parameter
import torch.nn.functional as F

class sp_GNN_ori(nn.Module):
    # the 
    def __init__(self, n_hid, n_feature, K, n_class):
        # K: order of cheb
        super(sp_GNN_ori, self).__init__()
        self.K = K
        self.feature_lin = nn.Linear(n_feature, n_hid, bias=True)
        # cheb poly coef
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()
        self.out_lin = nn.Linear(n_hid, n_class, bias=True)

    def reset_parameters(self):
        self.temp.data.fill_(0.0)
        self.temp.data[0]=1.0

    def forward(self, feature, L):
        batch_size = feature.shape[0]
        coe = self.temp
        X = self.feature_lin(feature)
        Tx_0 = X
        Tx_1 = torch.bmm(L, X)
        out = coe[0].repeat(batch_size, 1, 1) * Tx_0 + coe[1].repeat(batch_size, 1, 1) * Tx_1
        for i in range(2, self.K+1):
            Tx_2 = torch.bmm(L, Tx_1)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i].repeat(batch_size, 1, 1) * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        
        out = self.out_lin(out)
        return out


class sp_GNN(nn.Module):
    def __init__(self, K, num_classes, num_iter, n_hid, su, n_feat):
        # K: order of cheb
        super(sp_GNN, self).__init__()
        # cheby
        self.K = K
        # self.temp = Parameter(torch.Tensor(self.K + 1))
        # self.reset_parameters()

        self.num_iter_GNN = num_iter
        self.num_neuron = n_hid
        
        # MLP for nodes initialization
        self.fc1a=TimeDistributed(nn.Linear, n_feat, su)
        
        # MLP for factor nodes
        self.fc2a=TimeDistributed(nn.Linear, su, self.num_neuron)
        self.fc2b=TimeDistributed(nn.Linear, self.num_neuron, int(self.num_neuron/2))
        self.fc2c=TimeDistributed(nn.Linear, int(self.num_neuron/2), su)
                
        # MLP for GRU
        self.gru=TimeDistributed_GRU(torch.nn.GRUCell, su+2, self.num_neuron)
        
        # MLP for readout
        self.fc3a=TimeDistributed(nn.Linear, su, self.num_neuron)
        self.fc3b=TimeDistributed(nn.Linear, self.num_neuron, int(self.num_neuron/2))
        self.fc3c=TimeDistributed(nn.Linear, int(self.num_neuron/2), num_classes)
        
        self.fc4=TimeDistributed(nn.Linear, self.num_neuron, su)
        
        # Activation functions
        self.a1=nn.ReLU()

        # make the cheby coefficient data dependent
        self.coef_lin1 = TimeDistributed(nn.Linear, n_feat, self.num_neuron)
        self.coef_lin2 = TimeDistributed(nn.Linear, self.num_neuron, int(self.num_neuron/2))
        self.coef_lin3 = TimeDistributed(nn.Linear, int(self.num_neuron/2), self.K + 1)
        self.temp = Parameter(torch.Tensor(self.K + 1), requires_grad=True)
        self.coef = None

    def reset_parameters(self):
        # self.temp.data.fill_(0.0)
        self.temp.data[0]=1.0

    def cheby(self, X, L):
        batch_size = X.shape[0]
        # coe = self.temp # shared across batch
        Tx_0 = X
        Tx_1 = torch.bmm(L, X)
        # out = coe[0].repeat(batch_size, 1, 1) * Tx_0 + coe[1].repeat(batch_size, 1, 1) * Tx_1
        out = self.coef[:, 0, None, None] * Tx_0 + self.coef[:, 1, None, None] * Tx_1
        for i in range(2, self.K+1):
            Tx_2 = torch.bmm(L, Tx_1)
            Tx_2 = 2 * Tx_2 - Tx_0
            # out = out + coe[i].repeat(batch_size, 1, 1) * Tx_2
            out = out + (self.coef[:, i, None, None]/i) * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def forward(self, read_gru, gru_init, init_features, L, noise_info, x_hat_EP, variance_EP, GNN_iter):
        '''===========================custom layers========================'''
  
        def NN_iterations(init_feats, read_gru, gru_hidden, idx_iter, GNN_iter, L):
           
            if idx_iter == 0 and GNN_iter==0:
                x_init = init_feats
                batch_size = init_feats.shape[0]
                init_nodes = self.fc1a(x_init)# sample, nodes, features
                # compute cheby coefficient
                coef_hid = self.a1(self.coef_lin1(init_feats))
                coef_hid = self.a1(self.coef_lin2(coef_hid))
                coef_hid = self.coef_lin3(coef_hid)
                att = torch.bmm(coef_hid, self.temp.repeat(batch_size, 1).unsqueeze(-1))
                att_score = F.softmax(att, dim=1).squeeze()
                self.coef = torch.bmm(att_score.unsqueeze(-2), coef_hid).squeeze(-2)
            else:
                init_nodes = read_gru

            feat = self.a1(self.fc2a(init_nodes))
            feat = self.a1(self.fc2b(feat))
            feat = self.a1(self.fc2c(feat))

            sum_messages = self.cheby(feat, L)
             
            # GRU and outputs
            gru_feats_concat = torch.cat((sum_messages, torch.unsqueeze(x_hat_EP, 2), torch.unsqueeze(variance_EP, 2)), 2)
            gru_out = self.gru(gru_feats_concat, gru_hidden)
            read_gru = self.fc4(gru_out)

            return read_gru, gru_out
                    
        # Initializations
        init_feats = init_features
        gru_hidden = gru_init
        
        for idx_iter in range(self.num_iter_GNN):
            read_gru, gru_hidden = NN_iterations(init_feats, read_gru, gru_hidden, idx_iter, GNN_iter, L)
            
        R_out1 = self.fc3a(read_gru)
        R_out2 = self.fc3b(R_out1) 
        p_y_x = self.fc3c(R_out2)
        
        return p_y_x, read_gru, gru_hidden

class TimeDistributed(nn.Module): # input must be num_samples, nodes, features
    def __init__(self, module, *args):
        super(TimeDistributed, self).__init__()
        self.module = module(*args)

    def __call__(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1,x.size(-1))  # (samples * nodes, features)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(-1, x.size(1), y.size(-1))  # (samples, nodes, features)
        # else:
        #     y = y.view(-1, x.size(1), y.size(-1))  # (nodes, samples, output_size)
        return y

class TimeDistributed_GRU(nn.Module): # input must be num_samples, nodes, features
    def __init__(self, module, *args):
        super(TimeDistributed_GRU, self).__init__()
        self.module = module(*args)
        self.reset_parameters()

    def __call__(self, x,hx):

        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * nodes, input_size)
        hx_reshape = hx.contiguous().view(-1, hx.size(-1))  # (samples * nodes, input_size)

        y = self.module(x_reshape,hx_reshape) #400x128

        y = y.contiguous().view(-1, x.size(1), y.size(-1))  # (samples, nodes, output_size)
        return y
            
    def reset_parameters(self):
        # uniform(self.out_channels, self.weight)
        self.module.reset_parameters()
    



        



