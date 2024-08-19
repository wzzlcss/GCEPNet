import torch 
import numpy as np

def accuracy(out, j_indices_b):
    accuracy = (out == j_indices_b).sum().to(dtype=torch.float32)
    return accuracy.item()/out.numel()

def find_index(x_hat, constellation):
    x = x_hat.unsqueeze(dim=-1).expand(-1,-1, len(constellation))
    x = torch.pow(x - constellation, 2)
    x_indices = torch.argmin(x, dim=-1)
    return x_indices


