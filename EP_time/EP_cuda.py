import torch
from EP import EP


class EPNet(object):
    # GEPNet without GNN: for measuring time of the EP component
    def __init__(self, iter_GEP, beta, cons, device, dtype):  
        self.iter_GEP = iter_GEP
        self.beta = beta
        self.cons = torch.from_numpy(cons).to(dtype).to(device) # constellation set
        self.device = device
        self.dtype = dtype
        
    def forward(self, H, y, sigma2, k):        
        # Initialization
        user_num = H.shape[2] # num of x
        batch_size = H.shape[0]
        # (batch_size, x_num, 1)
        lamda = (torch.ones((batch_size, user_num, 1))*(3.0/(4**k - 1))).to(self.dtype).to(self.device)
        gamma = torch.zeros((batch_size, user_num, 1)).to(self.dtype).to(self.device)
        
        mean_ab = None
        var_ab = None
        p_GNN = None
        EP_model = EP(H, y, sigma2, user_num, self.cons, batch_size)
        
        # EP iteration
        for iteration in range(10):
            # (batch_size, x_num, x_num)
            diag_lamda = torch.zeros((batch_size, user_num, user_num)).to(self.dtype).to(self.device)
            #Perform EP: mean_ab, var_ab are from the cavity marginal
            mean_ab, var_ab, lamda, gamma = EP_model.performEP(self.beta, diag_lamda, p_GNN, mean_ab, var_ab, lamda, gamma, iteration)
            # Perform GNN
            # p_GNN, read_gru, gru = model(read_gru, gru, u_feats, edge, sigma2, mean_ab, var_ab, iteration)
            p_GNN = torch.rand((batch_size, user_num, self.cons.shape[0])).to(self.dtype).to(self.device)

        return p_GNN