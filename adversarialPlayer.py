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

np.random.seed(13)  #  for 2D systems, seed=13 gives two control angles, seed=10 gives multiple angles, seed=9 gives a single angle

dim_x, dim_z = 2, 2
N = 200  # time steps
batchSize = 1000
useCuda = False

enableSmartPlayers = True

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
tilde_z, tilde_x, processNoises, measurementNoises = GenMeasurements(N, batchSize, sysModel) # z: [N, batchSize, dim_z]
tilde_z, tilde_x, processNoises, measurementNoises = torch.tensor(tilde_z, dtype=torch.float), torch.tensor(tilde_x, dtype=torch.float), torch.tensor(processNoises, dtype=torch.float), torch.tensor(measurementNoises, dtype=torch.float)
if useCuda:
    tilde_z, tilde_x, processNoises, measurementNoises = tilde_z.cuda(), tilde_x.cuda(), processNoises.cuda(), measurementNoises.cuda()

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
tilde_e_k_given_k_minus_1 = tilde_x - tilde_x_est_f # k is the index so that at tilde_e_k_given_k_minus_1[0] we have tilde_e_0_given_minus_1
tilde_e_k_given_N_minus_1 = tilde_x - tilde_x_est_s

print(f'mean energy of tilde_x: ',{watt2dbm(calcTimeSeriesMeanEnergy(tilde_x).mean())},' [dbm]')

caligraphE_F_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_k_minus_1)
caligraphE_S_minus_1 = calcTimeSeriesMeanEnergyRunningAvg(tilde_e_k_given_N_minus_1)

delta_u, delta_caligraphE = 1e-3, 1e-3
adversarialPlayersToolbox = playersToolbox(pytorchEstimator, delta_u, delta_caligraphE, enableSmartPlayers)

# the next code checks the error expression used by the causal player
#tilde_e_k_given_k_minus_1_directCalc = adversarialPlayersToolbox.test_tilde_e_expression(tilde_x[0:1], filterStateInit, processNoises, measurementNoises, tilde_e_k_given_k_minus_1)

# No knowledge player:
u_0 = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
if useCuda:
    u_0 = u_0.cuda()
u_0 = noKnowledgePlayer(u_0)

print(f'mean energy of u_0: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_0).mean())},' [dbm]')

# Kalman filters:
z_0 = tilde_z + torch.matmul(H_transpose, u_0)
x_0_est_f, x_0_est_s = pytorchEstimator(z_0, filterStateInit)

e_R_0_k_given_k_minus_1 = tilde_x - x_0_est_f

caligraphE_F_0 = calcTimeSeriesMeanEnergyRunningAvg(e_R_0_k_given_k_minus_1)

# Smart players:
if enableSmartPlayers:
    u_1, u_2, u_3 = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float), torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float), torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
    if useCuda:
        u_1, u_2, u_3 = u_1.cuda(), u_2.cuda(), u_3.cuda()
    u_1, _ = noAccessPlayer(adversarialPlayersToolbox, u_1, torch.zeros_like(tilde_e_k_given_k_minus_1)) # tilde_e_k_given_k_minus_1 is given only for the window size calculation. It is legit
    u_3 = geniePlayer(adversarialPlayersToolbox, u_3, tilde_e_k_given_k_minus_1)
    u_2 = causalPlayer(adversarialPlayersToolbox, u_2, processNoises, tilde_x[0:1])

    print(f'mean energy of u_1: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_1).mean())},' [dbm]')
    print(f'mean energy of u_2: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_1).mean())}, ' [dbm]')
    print(f'mean energy of u_3: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_3).mean())}, ' [dbm]')

    z_1 = tilde_z + torch.matmul(H_transpose, u_1)
    x_1_est_f, x_1_est_s = pytorchEstimator(z_1, filterStateInit)

    e_R_1_k_given_k_minus_1 = tilde_x - x_1_est_f
    caligraphE_F_1 = calcTimeSeriesMeanEnergyRunningAvg(e_R_1_k_given_k_minus_1)

    z_2 = tilde_z + torch.matmul(H_transpose, u_2)
    x_2_est_f, x_2_est_s = pytorchEstimator(z_2, filterStateInit)

    e_R_2_k_given_k_minus_1 = tilde_x - x_2_est_f
    caligraphE_F_2 = calcTimeSeriesMeanEnergyRunningAvg(e_R_2_k_given_k_minus_1)

    z_3 = tilde_z + torch.matmul(H_transpose, u_3)
    x_3_est_f, x_3_est_s = pytorchEstimator(z_3, filterStateInit)

    e_R_3_k_given_k_minus_1 = tilde_x - x_3_est_f
    caligraphE_F_3 = calcTimeSeriesMeanEnergyRunningAvg(e_R_3_k_given_k_minus_1)


caligraphE_tVec = np.arange(0, N, 1)

# plotting:

caligraphE_F_minus_1, caligraphE_S_minus_1 = caligraphE_F_minus_1.detach().cpu().numpy(), caligraphE_S_minus_1.detach().cpu().numpy()
caligraphE_F_0 = caligraphE_F_0.detach().cpu().numpy()

if enableSmartPlayers:
    caligraphE_F_1 = caligraphE_F_1.detach().cpu().numpy()
    caligraphE_F_2 = caligraphE_F_2.detach().cpu().numpy()
    caligraphE_F_3 = caligraphE_F_3.detach().cpu().numpy()

trace_bar_Sigma = np.trace(pytorchEstimator.theoreticalBarSigma.cpu().numpy())
trace_bar_Sigma_S = np.trace(pytorchEstimator.theoreticalSmoothingSigma.cpu().numpy())

sigma_u_square = torch.tensor(1/dim_x, dtype=torch.float)
theoretical_caligraphE_F_0 = trace_bar_Sigma + sigma_u_square.cpu().numpy() * pytorchEstimator.normalizedNoKnowledgePlayerContribution.cpu().numpy()
theoretical_caligraphE_F_0_quadraticPart = sigma_u_square.cpu().numpy() * pytorchEstimator.normalizedNoKnowledgePlayerContribution.cpu().numpy()

enableDirctCalcsOnBlockVecs = False
if enableDirctCalcsOnBlockVecs:  # this shows that the gap for E(1) is legit
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

if enableSmartPlayers:
    theoretical_caligraphE_F_1 = trace_bar_Sigma + adversarialPlayersToolbox.theoretical_lambda_Xi_N_max.cpu().numpy()
    theoretical_upper_bound = trace_bar_Sigma + adversarialPlayersToolbox.theoretical_lambda_Xi_N_max.cpu().numpy() + 2*np.sqrt(adversarialPlayersToolbox.lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max.cpu().numpy()*trace_bar_Sigma)

# plotting batch 0:

batchIdx = 0

print(f'pure kalman performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_minus_1[-1])) - watt2dbm(np.mean(caligraphE_F_minus_1[-1]))} db')
print(f'no knowledge player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_0[-1])) - watt2dbm(np.mean(caligraphE_F_0[-1]))} db')
if enableSmartPlayers:
    print(f'no access player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_1[-1])) - watt2dbm(np.mean(caligraphE_F_1[-1]))} db')
    print(f'causal player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_2[-1])) - watt2dbm(np.mean(caligraphE_F_2[-1]))} db')
    print(f'genie player performance std w.r.t. mean: {watt2dbm(np.std(caligraphE_F_3[-1])) - watt2dbm(np.mean(caligraphE_F_3[-1]))} db')

caligraphE_F_minus_1_b = caligraphE_F_minus_1[:, batchIdx]  # watt
caligraphE_S_minus_1_b = caligraphE_S_minus_1[:, batchIdx]

caligraphE_F_minus_1_mean = np.mean(caligraphE_F_minus_1, axis=1)  # watt
#caligraphE_F_minus_1_mean = np.power(np.mean(np.sqrt(caligraphE_F_minus_1), axis=1), 2)  # watt
caligraphE_S_minus_1_mean = np.mean(caligraphE_S_minus_1, axis=1)  # watt

caligraphE_F_0_b = caligraphE_F_0[:, batchIdx]
caligraphE_F_0_mean = np.mean(caligraphE_F_0, axis=1)  # watt
#caligraphE_F_0_mean = np.power(np.mean(np.sqrt(caligraphE_F_0), axis=1), 2)  # watt

if enableSmartPlayers:
    caligraphE_F_1_b = caligraphE_F_1[:, batchIdx]
    caligraphE_F_1_mean = np.mean(caligraphE_F_1, axis=1)  # watt

    caligraphE_F_2_b = caligraphE_F_2[:, batchIdx]
    caligraphE_F_2_mean = np.mean(caligraphE_F_2, axis=1)  # watt

    caligraphE_F_3_b = caligraphE_F_3[:, batchIdx]
    caligraphE_F_3_mean = np.mean(caligraphE_F_3, axis=1)  # watt


plt.figure(figsize=(16,8))
plt.subplot(2, 2, 1)
plt.title('Absolute performance of players, specific game')

plt.plot(caligraphE_tVec, watt2dbm(theoretical_upper_bound * np.ones_like(caligraphE_tVec)), 'k--', label = r'theoretical upper bound')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_minus_1_b), 'g', label = r'empirical ${\cal E}^{(-1)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'g--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}\}$')

#plt.plot(caligraphE_tVec, watt2dbm(caligraphE_S_minus_1_b), label = r'empirical ${\cal E}^{(-1)}_{S,k}$')
#plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma_S * np.ones_like(caligraphE_S_minus_1_b)), '--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}^S\}$')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_b), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_tVec)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableSmartPlayers:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_b), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_tVec)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_2_b), color='brown', label=r'empirical ${\cal E}^{(2)}_{F,k}$')

    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_3_b), color='orange', label=r'empirical ${\cal E}^{(3)}_{F,k}$')

    #minY_absolute = np.min((watt2dbm(theoretical_upper_bound), np.min((watt2dbm(caligraphE_F_3_b), watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b), watt2dbm(caligraphE_F_1_b)))))
    #maxY_absolute = np.max((watt2dbm(theoretical_upper_bound), np.max((watt2dbm(caligraphE_F_3_b), watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b), watt2dbm(caligraphE_F_1_b)))))

marginAbsolute = 1 # db
#if enableSmartPlayers: plt.ylim([minY_absolute - marginAbsolute, maxY_absolute + marginAbsolute]
plt.legend()
plt.ylabel('dbm')
plt.grid()
bottom_221, top_221 = plt.ylim()
#plt.show()

plt.subplot(2, 2, 3)
plt.title('Players performance w.r.t pure filter, specific game')

plt.plot(caligraphE_tVec, watt2dbm(theoretical_upper_bound * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'k--', label = r'theoretical upper bound')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableSmartPlayers:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_2_b) - watt2dbm(caligraphE_F_minus_1_b), color='brown', label = r'empirical ${\cal E}^{(2)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_3_b) - watt2dbm(caligraphE_F_minus_1_b), color='orange', label = r'empirical ${\cal E}^{(3)}_{F,k}$')

    #minY_relative = np.min((watt2dbm(theoretical_upper_bound) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_3_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b)))
    #maxY_relative = np.max((watt2dbm(theoretical_upper_bound) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_3_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_0_b) - watt2dbm(caligraphE_F_minus_1_b), watt2dbm(caligraphE_F_1_b) - watt2dbm(caligraphE_F_minus_1_b)))

marginRelative = 5
#plt.legend()
plt.ylabel('db')
#if enableSmartPlayers: plt.ylim([minY_relative - marginRelative, maxY_relative + marginRelative])
plt.grid()
bottom_223, top_223 = plt.ylim()

plt.subplot(2, 2, 2)
plt.title('Absolute mean performance of players')

plt.plot(caligraphE_tVec, watt2dbm(theoretical_upper_bound * np.ones_like(caligraphE_tVec)), 'k--', label = r'theoretical upper bound')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_minus_1_mean), 'g', label = r'empirical ${\cal E}^{(-1)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'g--', label = r'theoretical $\operatorname{tr}\{\bar{\Sigma}\}$')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_mean), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_tVec)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableSmartPlayers:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_mean), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_tVec)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_2_mean), color='brown', label=r'empirical ${\cal E}^{(2)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_3_mean), color='orange', label=r'empirical ${\cal E}^{(3)}_{F,k}$')

#plt.legend()
plt.ylabel('dbm')
#if enableSmartPlayers: plt.ylim([minY_absolute - marginAbsolute, maxY_absolute + marginAbsolute])
bottom_222, top_222 = plt.ylim()
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Players mean performance w.r.t pure filter')

plt.plot(caligraphE_tVec, watt2dbm(theoretical_upper_bound * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'k--', label = r'theoretical upper bound')

plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_0_mean) - watt2dbm(caligraphE_F_minus_1_mean), 'b', label = r'empirical ${\cal E}^{(0)}_{F,k}$')
plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_0 * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'b--', label = r'theoretical $\operatorname{E}[{\cal E}_F^{(0)}]$')

if enableSmartPlayers:
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_1_mean) - watt2dbm(caligraphE_F_minus_1_mean), 'r', label = r'empirical ${\cal E}^{(1)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(theoretical_caligraphE_F_1 * np.ones_like(caligraphE_tVec)) - watt2dbm(trace_bar_Sigma * np.ones_like(caligraphE_tVec)), 'r--', label = r'theoretical ${\cal E}^{(1)}_{F,k}$')

    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_2_mean) - watt2dbm(caligraphE_F_minus_1_mean), color='brown', label = r'empirical ${\cal E}^{(2)}_{F,k}$')
    plt.plot(caligraphE_tVec, watt2dbm(caligraphE_F_3_mean) - watt2dbm(caligraphE_F_minus_1_mean), color='orange', label = r'empirical ${\cal E}^{(3)}_{F,k}$')

#plt.legend()
plt.ylabel('db')
#if enableSmartPlayers: plt.ylim([minY_relative - marginRelative, maxY_relative + marginRelative])
plt.grid()
bottom_224, top_224 = plt.ylim()

bottom_relative, top_relative = np.min((bottom_224, bottom_223)), np.max((top_224, top_223))
bottom_absolute, top_absolute = np.min((bottom_222, bottom_221)), np.max((top_222, top_221))
plt.ylim(bottom_relative, top_relative)

plt.subplot(2,2,3)
plt.ylim(bottom_relative, top_relative)
plt.subplot(2,2,1)
plt.ylim(bottom_absolute, top_absolute)
plt.subplot(2,2,2)
plt.ylim(bottom_absolute, top_absolute)

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
