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

#np.random.seed(21)  #  for 2D systems, seed=13 gives two control angles, seed=10 gives multiple angles, seed=9 gives a single angle

dim_x, dim_z = 2, 2
N = 200  # time steps
Ns_2_2N0_factor = 1000
batchSize = 100000
useCuda = False

enableNoAccessPlayer = True

# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)
'''
sysModel['F'] = 0.98*np.ones_like(sysModel['F'])
sysModel['R'] = 0.000001*np.ones_like(sysModel['R'])
sysModel['H'] = 1*np.ones_like(sysModel['H'])
sysModel['Q'] = 0.001*np.ones_like(sysModel['Q'])
'''

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)
if useCuda:
    pytorchEstimator = pytorchEstimator.cuda()
pytorchEstimator.eval()

# create time-series measurements (#time-series == batchSize):
tilde_z, tilde_x = GenMeasurements(N, batchSize, sysModel) # z: [N, batchSize, dim_z]
tilde_z, tilde_x = torch.tensor(tilde_z, dtype=torch.float), torch.tensor(tilde_x, dtype=torch.float)
if useCuda:
    tilde_z, tilde_x = tilde_z.cuda(), tilde_x.cuda()

# estimator init values:
filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(batchSize, dim_x, 1))
print(f'filter init mean error energy w.r.t trace(bar(sigma)): {watt2dbm(np.mean(np.power(np.linalg.norm(filterStateInit, axis=1), 2), axis=0)) - watt2dbm(np.trace(filter_P_init))} db')
filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False).contiguous()
#filterStateInit = tilde_x[0]

if useCuda:
    filterStateInit = filterStateInit.cuda()

#print(f'F = {sysModel["F"]}; H = {sysModel["H"]}; Q = {sysModel["Q"]}; R = {sysModel["R"]}')
H = torch.tensor(sysModel["H"], dtype=torch.float, requires_grad=False)
H_transpose = torch.transpose(H, 1, 0)
H_transpose = H_transpose.contiguous()
if useCuda:
    H_transpose = H_transpose.cuda()


tilde_x_est_f, tilde_x_est_s = pytorchEstimator(tilde_z, filterStateInit)
# tilde_x_est_f = hat_x_k_plus_1_given_k
tilde_e_k_given_k_minus_1 = tilde_x_est_f - tilde_x # k is the index so that at tilde_e_k_given_k_minus_1[0] we have tilde_e_0_given_minus_1
tilde_e_k_given_N_minus_1 = tilde_x_est_s - tilde_x

print(f'mean energy of tilde_x: ',{watt2dbm(calcTimeSeriesMeanEnergy(tilde_x).mean())},' [dbm]')

delta_u, delta_caligraphE = 1e-3, 1e-3
adversarialPlayersToolbox = playersToolbox(pytorchEstimator, delta_u, delta_caligraphE, enableNoAccessPlayer, Ns_2_2N0_factor)

# No knowledge player:
u_0 = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
if useCuda:
    u_0 = u_0.cuda()
u_0 = noKnowledgePlayer(u_0)

print(f'mean energy of u_0: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_0).mean())},' [dbm]')

# No access player:
if enableNoAccessPlayer:
    u_1 = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
    if useCuda:
        u_1 = u_1.cuda()
    u_1, _ = noAccessPlayer(adversarialPlayersToolbox, u_1, tilde_e_k_given_k_minus_1) # tilde_e_k_given_k_minus_1 is given only for the window size calculation. It is legit

    print(f'mean energy of u_1: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_1).mean())},' [dbm]')

    z_1 = tilde_z + torch.matmul(H_transpose, u_1)
    x_1_est_f, x_1_est_s = pytorchEstimator(z_1, filterStateInit)

    e_R_1_k_given_k_minus_1 = x_1_est_f - tilde_x
    caligraphE_F_1 = calcTimeSeriesMeanEnergyRunningAvg(e_R_1_k_given_k_minus_1)

# Kalman filters:
z_0 = tilde_z + torch.matmul(H_transpose, u_0)
x_0_est_f, x_0_est_s = pytorchEstimator(z_0, filterStateInit)

e_R_0_k_given_k_minus_1 = x_0_est_f - tilde_x

caligraphE_F_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_k_minus_1)
caligraphE_S_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_N_minus_1)

caligraphE_F_0 = calcTimeSeriesMeanEnergyRunningAvg(e_R_0_k_given_k_minus_1)

caligraphE_tVec = np.arange(0, N, 1)

# plotting:

caligraphE_F_minus_1, caligraphE_S_minus_1 = caligraphE_F_minus_1.detach().cpu().numpy(), caligraphE_S_minus_1.detach().cpu().numpy()
caligraphE_F_0 = caligraphE_F_0.detach().cpu().numpy()

if enableNoAccessPlayer:
    caligraphE_F_1 = caligraphE_F_1.detach().cpu().numpy()

trace_bar_Sigma = np.trace(pytorchEstimator.theoreticalBarSigma.cpu().numpy())
trace_bar_Sigma_S = np.trace(pytorchEstimator.theoreticalSmoothingSigma.cpu().numpy())

sigma_u_square = torch.tensor(1/dim_x, dtype=torch.float)
theoretical_caligraphE_F_0 = trace_bar_Sigma + sigma_u_square.cpu().numpy() * pytorchEstimator.normalizedNoKnowledgePlayerContribution.cpu().numpy()
theoretical_caligraphE_F_0_quadraticPart = sigma_u_square.cpu().numpy() * pytorchEstimator.normalizedNoKnowledgePlayerContribution.cpu().numpy()

enableDirctCalcsOnBlockVecs = False
if enableDirctCalcsOnBlockVecs:
    # no knowledge player calculation directly from block vectors:
    # no knowledge player expected gap between theoretical and empirical:
    blockVec_tilde_e_full = tilde_e_k_given_k_minus_1.permute(1, 0, 2, 3).reshape(batchSize, N * dim_x, 1)
    u_0_blockVec = u_0.permute(1, 0, 2, 3).reshape(batchSize, N * dim_x, 1)
    caligraphE_directCalc, caligraphE_directCalc_linearPart, caligraphE_directCalc_noPlayerPart, caligraphE_directCalc_quadraticPart = adversarialPlayersToolbox.compute_caligraphE(u_0_blockVec, blockVec_tilde_e_full)
    caligraphE_directCalc, caligraphE_directCalc_linearPart, caligraphE_directCalc_noPlayerPart, caligraphE_directCalc_quadraticPart = caligraphE_directCalc.mean().cpu().numpy(), caligraphE_directCalc_linearPart.mean().cpu().numpy(), caligraphE_directCalc_noPlayerPart.mean().cpu().numpy(), caligraphE_directCalc_quadraticPart.mean().cpu().numpy()
    print(f'No knowledge player empiric pure part w.r.t theoretic pure: {watt2db(caligraphE_directCalc_noPlayerPart/trace_bar_Sigma)} db')
    print(f'No knowledge player empiric quadratic part w.r.t theoretic quadratic part: {watt2db(caligraphE_directCalc_quadraticPart/theoretical_caligraphE_F_0_quadraticPart)} db')
    print(f'No knowledge player empiric performance from block vectors: {watt2dbm(caligraphE_directCalc)} dbm')
    print(f'No knowledge player empiric performance linear part: {watt2dbm(caligraphE_directCalc_linearPart)} dbm')
    print(f'No knowledge player empiric performance quadratic part: {watt2dbm(caligraphE_directCalc_quadraticPart)} dbm')

if enableNoAccessPlayer:
    theoretical_caligraphE_F_1 = trace_bar_Sigma + adversarialPlayersToolbox.theoretical_lambda_Xi_N_max.cpu().numpy()

# plotting batch 0:

batchIdx = 0

print(f'pure kalman performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_minus_1[-1])) - watt2dbm(np.mean(caligraphE_F_minus_1[-1]))} db')
print(f'no knowledge player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_0[-1])) - watt2dbm(np.mean(caligraphE_F_0[-1]))} db')
if enableNoAccessPlayer:
    print(f'no access player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_1[-1])) - watt2dbm(np.mean(caligraphE_F_1[-1]))} db')

caligraphE_F_minus_1_b = caligraphE_F_minus_1[:, batchIdx]  # watt
caligraphE_S_minus_1_b = caligraphE_S_minus_1[:, batchIdx]

caligraphE_F_minus_1_mean = np.mean(caligraphE_F_minus_1, axis=1)  # watt
#caligraphE_F_minus_1_mean = np.power(np.mean(np.sqrt(caligraphE_F_minus_1), axis=1), 2)  # watt
caligraphE_S_minus_1_mean = np.mean(caligraphE_S_minus_1, axis=1)  # watt

caligraphE_F_0_b = caligraphE_F_0[:, batchIdx]
if enableNoAccessPlayer:
    caligraphE_F_1_b = caligraphE_F_1[:, batchIdx]
    caligraphE_F_1_mean = np.mean(caligraphE_F_1, axis=1)  # watt

caligraphE_F_0_mean = np.mean(caligraphE_F_0, axis=1)  # watt
#caligraphE_F_0_mean = np.power(np.mean(np.sqrt(caligraphE_F_0), axis=1), 2)  # watt


plt.figure(figsize=(16,8))
plt.subplot(2, 2, 1)
plt.title('Absolute performance of players, specific game')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_minus_1_b), 'g', label = r'empirical ${\cal E}^{(-1)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_b)), 'g--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}\}$')

#plt.plot(caligraphE_tVec, watt2dbm(caligraphE_S_minus_1_b), label = r'empirical ${\cal E}^{(-1)}_{S,k}$')
#plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma_S * np.ones_like(caligraphE_S_minus_1_b)), '--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}^S\}$')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_b), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_F_0_b)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableNoAccessPlayer:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_b), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_F_1_b)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    minY_absolute = np.min((watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b), watt2dbm(caligraphE_F_1_b)))
    maxY_absolute = np.max((watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b), watt2dbm(caligraphE_F_1_b)))

marginAbsolute = 1 # db
if enableNoAccessPlayer: plt.ylim([minY_absolute - marginAbsolute, maxY_absolute + marginAbsolute])
plt.legend()
plt.ylabel('dbm')
plt.grid()
#plt.show()

plt.subplot(2, 2, 3)
plt.title('Players performance w.r.t pure filter, specific game')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_F_0_b)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_b)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableNoAccessPlayer:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_F_1_b)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_b)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    minY_relative = np.min((watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b)))
    maxY_relative = np.max((watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b)))

marginRelative = 1
plt.legend()
plt.ylabel('db')
if enableNoAccessPlayer: plt.ylim([minY_relative - marginRelative, maxY_relative + marginRelative])
plt.grid()

plt.subplot(2, 2, 2)
plt.title('Absolute mean performance of players')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_minus_1_mean), 'g', label = r'empirical ${\cal E}^{(-1)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_mean)), 'g--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}\}$')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_mean), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_F_0_mean)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableNoAccessPlayer:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_mean), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_F_1_mean)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

plt.legend()
plt.ylabel('dbm')
if enableNoAccessPlayer: plt.ylim([minY_absolute - marginAbsolute, maxY_absolute + marginAbsolute])
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Players mean performance w.r.t pure filter')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_mean) - watt2dbm(caligraphE_F_minus_1_mean), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_F_0_mean)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_mean)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableNoAccessPlayer:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_mean) - watt2dbm(caligraphE_F_minus_1_mean), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_F_1_mean)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_F_minus_1_mean)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

plt.legend()
plt.ylabel('db')
if enableNoAccessPlayer: plt.ylim([minY_relative - marginRelative, maxY_relative + marginRelative])
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
