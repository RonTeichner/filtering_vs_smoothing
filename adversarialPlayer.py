import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db, volt2dbm, watt2dbm, calc_tildeC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

#np.random.seed(13)

dim_x, dim_z = 5, 3
N = 100000  # time steps
batchSize = 1
useCuda = False

# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)
'''
sysModel['F'] = 0.98*np.ones_like(sysModel['F'])
sysModel['R'] = 0.000001*np.ones_like(sysModel['R'])
sysModel['H'] = 1*np.ones_like(sysModel['H'])
sysModel['Q'] = 0.001*np.ones_like(sysModel['Q'])
'''
# create time-series measurements (#time-series == batchSize):
tilde_z, tilde_x = GenMeasurements(N, batchSize, sysModel) # z: [N, batchSize, dim_z]
tilde_z, tilde_x = torch.tensor(tilde_z, dtype=torch.float), torch.tensor(tilde_x, dtype=torch.float)
if useCuda:
    tilde_z, tilde_x = tilde_z.cuda(), tilde_x.cuda()

# estimator init values:
filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False).contiguous()
#filterStateInit = tilde_x[0]

if useCuda:
    filterStateInit = filterStateInit.cuda()

print(f'F = {sysModel["F"]}; H = {sysModel["H"]}; Q = {sysModel["Q"]}; R = {sysModel["R"]}')
H = torch.tensor(sysModel["H"], dtype=torch.float, requires_grad=False)
H_transpose = torch.transpose(H, 1, 0)
H_transpose = H_transpose.contiguous()
if useCuda:
    H_transpose = H_transpose.cuda()

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)
if useCuda:
    pytorchEstimator = pytorchEstimator.cuda()
pytorchEstimator.eval()

tilde_x_est_f, tilde_x_est_s = pytorchEstimator(tilde_z, filterStateInit)
# tilde_x_est_f = hat_x_k_plus_1_given_k

# No knowledge player:
sigma_u_square = 8*torch.tensor(1/dim_x, dtype=torch.float)
u_0 = noKnowledgePlayer(N, batchSize, dim_x, sigma_u_square)
if useCuda:
    u_0 = u_0.cuda()

print(f'mean energy of tilde_x: ',{watt2dbm(calcTimeSeriesMeanEnergy(tilde_x))},' [dbm]')
print(f'mean energy of u_0: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_0))},' [dbm]')

z_0 = tilde_z + torch.matmul(H_transpose, u_0)

x_0_est_f, x_0_est_s = pytorchEstimator(z_0, filterStateInit)

tilde_e_k_given_k_minus_1 = tilde_x_est_f - tilde_x # k is the index so that at tilde_e_k_given_k_minus_1[0] we have tilde_e_0_given_minus_1
tilde_e_k_given_N_minus_1 = tilde_x_est_s - tilde_x
e_R_0_k_given_k_minus_1 = x_0_est_f - tilde_x

caligraphE_F_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_k_minus_1)
caligraphE_S_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_N_minus_1)

caligraphE_F_0 = calcTimeSeriesMeanEnergyRunningAvg(e_R_0_k_given_k_minus_1)

caligraphE_tVec = np.arange(0, N, 1)

# plotting:

caligraphE_F_minus_1 = caligraphE_F_minus_1.detach().cpu().numpy()
caligraphE_F_0 = caligraphE_F_0.detach().cpu().numpy()

trace_bar_Sigma = np.trace(pytorchEstimator.theoreticalBarSigma.cpu().numpy())
trace_bar_Sigma_S = np.trace(pytorchEstimator.theoreticalSmoothingSigma.cpu().numpy())

theoretical_caligraphE_F_0 = trace_bar_Sigma + sigma_u_square.cpu().numpy() * pytorchEstimator.normalizedNoKnowledgePlayerContribution.cpu().numpy()

# plotting batch 0:
batchIdx = 0

caligraphE_F_minus_1_b = caligraphE_F_minus_1[:, batchIdx]
caligraphE_S_minus_1_b = caligraphE_S_minus_1[:, batchIdx]

caligraphE_F_0_b = caligraphE_F_0[:, batchIdx]

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_minus_1_b), label = r'empirical ${\cal E}^{(-1)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_b)), '--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}\}$')

#plt.plot(caligraphE_tVec, watt2dbm(caligraphE_S_minus_1_b), label = r'empirical ${\cal E}^{(-1)}_{S,k}$')
#plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma_S * np.ones_like(caligraphE_S_minus_1_b)), '--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}^S\}$')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_b), label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_F_0_b)), '--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

#minTheoretic = np.min((watt2dbm(trace_bar_Sigma), watt2dbm(trace_bar_Sigma_S), watt2dbm(theoretical_caligraphE_F_0)))
#maxTheoretic = np.max((watt2dbm(trace_bar_Sigma), watt2dbm(trace_bar_Sigma_S), watt2dbm(theoretical_caligraphE_F_0)))

minTheoretic = np.min((watt2dbm(trace_bar_Sigma), watt2dbm(theoretical_caligraphE_F_0)))
maxTheoretic = np.max((watt2dbm(trace_bar_Sigma), watt2dbm(theoretical_caligraphE_F_0)))

margin = 1 # db
plt.ylim([minTheoretic - margin, maxTheoretic + margin])
plt.legend()
plt.ylabel('dbm')
plt.grid()
plt.show()

'''
# tilde_e_k_given_k_minus_1.detach().cpu().numpy()
tilde_x = tilde_x.detach().cpu().numpy()[:,0,0,0]
tilde_e_k_given_k_minus_1 = tilde_e_k_given_k_minus_1.detach().cpu().numpy()[:,0,0,0]
tilde_z = tilde_z.detach().cpu().numpy()[:,0,0,0]
tilde_x_est_f = tilde_x_est_f.detach().cpu().numpy()[:,0,0,0]
plt.plot(tilde_x, label = r'$\tilde{x}$')
plt.plot(tilde_z, label = r'$z$')
plt.plot(tilde_x_est_f, label = r'$\hat{\tilde{x}}$')
plt.plot(tilde_e_k_given_k_minus_1, label = r'$\tilde{e}$')
plt.plot(np.cumsum(tilde_e_k_given_k_minus_1), label = r'$\sum\tilde{e}$')
plt.legend()
plt.grid()
plt.show()
'''
