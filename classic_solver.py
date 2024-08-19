import numpy as np
import torch
import cvxpy as cp
import torch.nn.functional as F
from helper import *

def mmse(y, H, sigma2, k):
	two_NT = int(H.shape[-1])
	H_t = H.permute(0,2,1)
	Hty = torch.einsum(('ijk,ik->ij'), (H_t, y))
	HTH = torch.matmul(H_t, H)
	# HtHinv = torch.inverse(HTH + ((torch.pow(sigma, 2)/2.0).view(-1, 1, 1))*torch.eye(n=two_NT).expand(size=HTH.shape))
	temp = ((3.0 * sigma2/(4**k - 1)) * torch.eye(two_NT)).expand(size=HTH.shape)
	HtHinv = torch.inverse(HTH + temp)
	x = torch.einsum(('ijk,ik->ij'), (HtHinv, Hty))
	return x

def zf_detector(y, H):
	return np.matmul(np.linalg.pinv(H), y)

def symbol_detection(y, constellation):
    # round to the nearest integer, return the index of the signal_list
    return np.expand_dims(np.argmin(np.abs(y-np.expand_dims(constellation, 0)), axis=1),1)

def babai(hBatch, yBatch, constellation, p=None):
	# if p is specified, we use regularized babai
	batch_size = hBatch.shape[0]
	n = hBatch.shape[-1]
	xBatch = np.zeros((batch_size, n))
	for i in range(batch_size):
		H = hBatch[i]; y = yBatch[i]
		if p is not None:
			H = np.concatenate((H, p * np.eye(n)), axis = 0)
			y = np.concatenate((y, np.zeros((n, 1))))
		Q, R = np.linalg.qr(H, mode='reduced')
		D = np.diag(np.sign(np.diag(R)))
		Q = Q @ D
		R = D @ R
		y1 = Q.T @ y
		x = [0]*n
		for k in range(n-1, -1, -1): 
			if k == n-1: 
				c = y1[k]/R[k, k]
			else: 
				c = (y1[k] - R[k, k+1:n] @ x[k+1:n]) / R[k,k]			
			symbol_index = np.argmin(np.power(c - constellation, 2))
			x[k] = constellation[symbol_index]
		xBatch[i] = x
	return xBatch

# expectation propagation
def EP(y_b, H_b, sigma2, num_iter, Nt, constellation, k):
	# sigma2 is the variance
	user_num = Nt*2
	lamda_init = np.ones((H_b.shape[0], Nt*2))*(3.0/(4**k - 1))
	gamma_init = np.zeros((H_b.shape[0], Nt*2))
	sigma2 = np.mean(sigma2) # the mean of variance
	H = H_b
	y = y_b
	constellation_expanded = np.expand_dims(constellation, axis=1)
	constellation_expanded= np.repeat(constellation_expanded[None, ...], H.shape[0], axis=0) #[batch_size, num_symbol, 1]

	def calculate_mean_var(pyx, constellation_expanded):
		constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
		mean = np.matmul(pyx, constellation_expanded)
		var = np.square(np.abs(constellation_expanded_transpose - mean))
		var = np.multiply(pyx, var) 
		var = np.sum(var, axis=2)		
		return np.squeeze(mean), var

	def calculate_pyx(mean, var, constellation_expanded):
		constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), user_num, axis=1)
		arg_1 = np.square(np.abs(constellation_expanded_transpose - np.expand_dims(mean,2)))
		log_pyx = (-1 * arg_1)/(2*np.expand_dims(var,2))
		log_pyx = log_pyx - np.expand_dims(np.max(log_pyx, 2),2)
		p_y_x = np.exp(log_pyx)
		p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=2),2) + np.finfo(float).eps)
		return p_y_x

	def LMMSE( H, y, sigma2, lamda, gamma):
		HtH = np.matmul(np.transpose(H, [0,2,1]), H)
		Hty = np.squeeze(np.matmul(np.transpose(H,[0,2,1]), np.expand_dims(y,2)))
		diag_lamda = np.zeros((HtH.shape[0], user_num, user_num))
		np.einsum('ijj->ij', diag_lamda)[...] = lamda
		var = np.linalg.inv(HtH + diag_lamda * sigma2)
		mean = (Hty) + gamma * sigma2
		mean = np.matmul(var, np.expand_dims(mean,2)) # eq 29
		var = var * sigma2
		return mean, var

	lamda = lamda_init
	gamma = gamma_init
	for iteration in range(num_iter):					
		mean_mmse, var_mmse = LMMSE(H, y, sigma2, lamda, gamma)
		# Calculating mean and variance of P_y_x
		diag_mmse=np.diagonal(var_mmse, axis1=1, axis2=2)
		var_ab = (diag_mmse/ (1 - diag_mmse*lamda )) + np.finfo(float).eps	# eq 31
		# var_ab = 1 / (1/np.diagonal(var_mmse, axis1=1, axis2=2) - lamda )
		mean_ab = (np.squeeze(mean_mmse)/np.diagonal(var_mmse, axis1=1, axis2=2) - gamma) 
		mean_ab = var_ab * mean_ab # eq 32
		# Calculating P_y_x
		p_y_x_ab = calculate_pyx(mean_ab, var_ab, constellation_expanded)
		# Calculating mean and variance of \hat{P}_x_y
		mean_b, var_b = calculate_mean_var(p_y_x_ab, constellation_expanded)
		var_b = np.clip(var_b, 1e-13, None)
		# Calculating new lamda and gamma
		lamda_new = ((var_ab-var_b) / var_b )/ var_ab # eq35	
		# lamda_new = 1/ var_b - 1/var_ab
		gamma_new = mean_b /var_b - mean_ab/ var_ab # eq 36
		# Avoiding negative lamda and gamma
		if np.any(lamda_new < 0):
			indices = np.where(lamda_new < 0)
			lamda_new[indices]=lamda[indices]
			gamma_new[indices]=gamma[indices]
		# Appliying updating weight
		lamda = lamda*0.7 + lamda_new*0.3
		gamma = gamma*0.7 + gamma_new*0.3
	# SER = self.calc_perf(mean_b)
	# use mean_b to give the detection result
	return mean_b

def oamp(H, y, sigma2, constellation, layer):
	constellation_expanded = np.expand_dims(constellation, axis=1)
	constellation_expanded = np.repeat(constellation_expanded[None, ...], H.shape[0], axis=0) #[batch_size, num_symbol, 1]
	# sigma2 shoud be the variance of the complex noise
	# H, y being numpy array
	Nt = int(H.shape[2]/2)
	Nr = int(H.shape[1]/2)
	batch_size = H.shape[0]
	# initialize
	xhatt = np.zeros((batch_size, 2 * Nt, 1))
	rt = y
	HtH = np.matmul(np.transpose(H, [0,2,1]), H)
	HHt = np.matmul(H, np.transpose(H, [0,2,1]))
	for layeri in range(layer):
		# oamp iteration
		# calculate v2t
		num = np.sum(np.square(rt), axis=1) - Nt * sigma2
		den = np.trace(HtH, axis1=1, axis2=2) + np.finfo(float).eps
		v2t = num / np.expand_dims(den, 1)
		v2t = np.maximum(v2t, 1e-9 * np.ones((batch_size, 1)))
		# calculate Whatt
		diag_batch = np.repeat(np.eye(2*Nr)[np.newaxis, :, :], batch_size, axis=0)
		diag_value = np.ones((batch_size, 1, 1)) * (0.5 * sigma2)
		temp = np.linalg.inv(np.expand_dims(v2t, 2) * HHt + diag_batch * diag_value)
		Whatt = np.expand_dims(v2t, 2) * np.matmul(np.transpose(H, [0,2,1]), temp)
		# calculate Wt
		WhattH_tr = np.trace(np.matmul(Whatt, H), axis1=1, axis2=2) # (batch_size, )
		coef = (2 * Nr) / WhattH_tr # (batch_size, )
		Wt = coef.reshape(batch_size, 1, 1) * Whatt
		#
		zt = xhatt + np.matmul(Wt, rt)
		Bt = diag_batch - np.matmul(Wt, H)
		# calculate tau
		BBt = np.matmul(Bt, np.transpose(Bt, [0,2,1]))
		WWt = np.matmul(Wt, np.transpose(Wt, [0,2,1]))
		BBt_tr = np.trace(BBt, axis1=1, axis2=2)
		WWt_tr = np.trace(WWt, axis1=1, axis2=2)
		term1 = np.expand_dims(1.0/(2 * Nr) * BBt_tr, 1) * v2t
		term2 = np.expand_dims(1.0/(4 * Nr) * WWt_tr, 1) * sigma2
		tau2t = term1 + term2
		# 
		constellation_expanded_transpose = np.repeat(constellation_expanded.transpose(0,2,1), 2 * Nr, axis=1)
		arg_1 = np.square(np.abs(constellation_expanded_transpose - zt))
		log_pyx = (-1 * arg_1)/(2.0 * np.expand_dims(tau2t, 2))
		log_pyx = log_pyx - np.expand_dims(np.max(log_pyx, 2), 2)
		p_y_x = np.exp(log_pyx)
		p_y_x = p_y_x/(np.expand_dims(np.sum(p_y_x, axis=2),2) + np.finfo(float).eps)
		xhatt = np.matmul(p_y_x, constellation_expanded)
		rt = y - np.matmul(H, xhatt)
	return xhatt


	





















