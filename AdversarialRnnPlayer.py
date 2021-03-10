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

fileName = 'sys2D_secondTry'
useCuda = True
if useCuda:
    device = 'cuda'
else:
    device = 'cpu'

batchSize = 100
num_layers, hidden_size = 2, 100
dp = False

lowThrLr = 1e-6

playerType = 'Causal'  # {'NoAccess', 'Causal', 'Genie'}
if playerType == 'NoAccess':
    p = 1
elif playerType == 'Causal':
    p = 2
elif playerType == 'Genie':
    p = 3

savedList = pickle.load(open(fileName + '_final_' + '.pt', "rb"))
sysModel, bounds_N, currentFileName_N, bounds_N_plus_m, currentFileName_N_plus_m, bounds_N_plus_2m, currentFileName_N_plus_2m, mistakeBound, delta_trS, gapFromInfBound = savedList
savedList = pickle.load(open(currentFileName_N_plus_2m, "rb"))
N = savedList[2]

print(f'bounds file loaded; bounds were calculated for N = {N}')
print(f'no player bound is {watt2dbm(bounds_N_plus_2m[0])} dbm')
print(f'no knowledge bound is {watt2dbm(bounds_N_plus_2m[1])} dbm; {watt2dbm(bounds_N_plus_2m[1]) - watt2dbm(bounds_N_plus_2m[0])} db')
print(f'no access bound is {watt2dbm(bounds_N_plus_2m[2])} dbm; {watt2dbm(bounds_N_plus_2m[2]) - watt2dbm(bounds_N_plus_2m[0])} db')
print(f'causal bound is {watt2dbm(bounds_N_plus_2m[3])} dbm; {watt2dbm(bounds_N_plus_2m[3]) - watt2dbm(bounds_N_plus_2m[0])} db')
print(f'genie bound is {watt2dbm(bounds_N_plus_2m[4])} dbm; {watt2dbm(bounds_N_plus_2m[4]) - watt2dbm(bounds_N_plus_2m[0])} db')

theoreticalPlayerImprovement = [watt2dbm(bounds_N_plus_2m[2]) - watt2dbm(bounds_N_plus_2m[0]), watt2dbm(bounds_N_plus_2m[3]) - watt2dbm(bounds_N_plus_2m[0]), watt2dbm(bounds_N_plus_2m[4]) - watt2dbm(bounds_N_plus_2m[0])]  # [noAccess, Causal, Genie]

dim_x, dim_z = sysModel['F'].shape[0], sysModel['H'].shape[1]
Q_cholesky, R_cholesky = torch.tensor(np.linalg.cholesky(sysModel['Q']), dtype=torch.float, device=device), torch.tensor(np.linalg.cholesky(sysModel['R']), dtype=torch.float, device=device)

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=False, useCuda=useCuda)

# class definition
class RNN_Adversarial(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN_Adversarial, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup RNN layer
        #self.Adv_rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)
        self.Adv_rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def normalizeControlEnergy(self, u_NxN):
        N, batchSize, useCuda = u_NxN.shape[0], u_NxN.shape[1], u_NxN.is_cuda
        u_NxN = u_NxN.reshape(N, batchSize, N, -1)  # shape: [N, batchSize, N, dim_x]
        dim_x = u_NxN.shape[3]

        if useCuda:
            totalLeftEnergy = N*torch.ones(batchSize, dtype=torch.float).cuda()
            u_N = torch.zeros(N, batchSize, dim_x, dtype=torch.float).cuda()
            N_tensor = torch.tensor(np.sqrt(N), dtype=torch.float).cuda()
        else:
            totalLeftEnergy = N * torch.ones(batchSize, dtype=torch.float)
            u_N = torch.zeros(N, batchSize, dim_x, dtype=torch.float)
            N_tensor = torch.tensor(np.sqrt(N), dtype=torch.float)

        for k in range(N):
            u_N_at_k = u_NxN[k].permute(1, 0, 2)  # shape: [N, batchSize, dim_x]
            u_N_at_k_energyPerBatch = u_N_at_k.pow(2).sum(dim=2).sum(dim=0)
            u_N_at_k_unitEnergy = torch.div(u_N_at_k, torch.sqrt(u_N_at_k_energyPerBatch)[None, :, None].repeat(N, 1, dim_x))
            u_N_at_k_legitEnergy = torch.multiply(u_N_at_k_unitEnergy, N_tensor) # now u_N_at_k_legitEnergy has legit energy of N

            # u_N_at_k_legitEnergy[:, 10].pow(2).sum()  #  this is the energy in a single batch

            u_N_at_k_futureActions = u_N_at_k_legitEnergy[k:]  # shape: [N-k, batchSize, dim_x]
            u_N_at_k_futureActions_energyPerBatch = u_N_at_k_futureActions.pow(2).sum(dim=2).sum(dim=0)
            u_N_at_k_futureActions_unitEnergy = torch.div(u_N_at_k_futureActions, torch.sqrt(u_N_at_k_futureActions_energyPerBatch)[None, :, None].repeat(N-k, 1, dim_x))
            u_N_at_k_futureActions_leftEnergy = torch.multiply(u_N_at_k_futureActions_unitEnergy, torch.sqrt(totalLeftEnergy[None, :, None].repeat(N-k, 1, dim_x)))

            # u_N_at_k_futureActions_leftEnergy[:, 10].pow(2).sum()  #  this is the energy in a single batch

            nextAction = u_N_at_k_futureActions_leftEnergy[0]
            u_N[k] = nextAction
            totalLeftEnergy = totalLeftEnergy - torch.pow(torch.linalg.norm(nextAction, dim=1), 2)
        return u_N

    def forward(self, processNoiseBlockVec, measurementNoiseBlockVec):
        # processNoiseBlockVec, measurementNoiseBlockVec shapes: [N, batchSize, dim_x*N]
        controlHiddenDim, hidden = self.Adv_rnn(torch.cat((processNoiseBlockVec, measurementNoiseBlockVec), dim=2))
        # controlHiddenDim shape: [N, batchSize, hidden_dim]
        control_dim_x = self.linear(controlHiddenDim)
        # control_dim_x shape: [N, batchSize, dim_x*N]
        control_dim_x = self.normalizeControlEnergy(control_dim_x)

        return control_dim_x[:, :, :, None], hidden

# Define player
print("Build RNN player model ...")
player = RNN_Adversarial(input_dim=N*(dim_x + dim_z), hidden_dim=hidden_size, output_dim=N*dim_x, num_layers=num_layers).to(device=device)

optimizer = optim.Adam(player.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, threshold=1e-6)

H = torch.tensor(sysModel["H"], dtype=torch.float, requires_grad=False)
H_transpose = torch.transpose(H, 1, 0)
H_transpose = H_transpose.contiguous()
if useCuda:
    H_transpose = H_transpose.cuda()

epoch = -1
displayEvery_n_epochs = 100
startTime = time.time()
nEpochsWithNoSaveThr = 1000
saveThr = 1e-4
lowThrLr = 1e-6
filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
hidden = torch.zeros(num_layers, batchSize, hidden_size, dtype=torch.float, device=device)
bestPerformanceVsNoPlayer = -np.inf
playerFileName = fileName + '_' + playerType + '.pt'
while True:
    epoch += 1
    optimizer.zero_grad()
    pytorchEstimator.zero_grad()

    # create pure measurements:
    tilde_z, tilde_x, processNoises, measurementNoises = GenMeasurements(N, batchSize, sysModel, startAtZero=False, dp=dp)  # z: [N, batchSize, dim_z]
    tilde_z, tilde_x, processNoises, measurementNoises = torch.tensor(tilde_z, dtype=torch.float, device=device), torch.tensor(tilde_x, dtype=torch.float, device=device), torch.tensor(processNoises, dtype=torch.float, device=device), torch.tensor(measurementNoises, dtype=torch.float, device=device)

    # knowledge gate:
    processNoisesKnown2Player, measurementNoisesKnown2Player = knowledgeGate(Q_cholesky, R_cholesky, playerType, processNoises, measurementNoises, device)
    # processNoisesKnown2Player  shape: [N, batchSize, N, dim_x, 1]
    # measurementNoisesKnown2Player  shape: [N, batchSize, N, dim_z, 1]

    # Adversarial player:
    u_N, _ = player(processNoisesKnown2Player[:, :, :, :, 0].reshape(N, batchSize, -1), measurementNoisesKnown2Player[:, :, :, :, 0].reshape(N, batchSize, -1))

    #  u_N.pow(2).sum(dim=2).sum(dim=0)  # this shows the energy used by the player at every batch

    # estimator init values:
    filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(batchSize, dim_x, 1))
    if dp: print(f'filter init mean error energy w.r.t trace(bar(sigma)): {watt2dbm(np.mean(np.power(np.linalg.norm(filterStateInit, axis=1), 2), axis=0)) - watt2dbm(np.trace(filter_P_init))} db')
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
    # filterStateInit = tilde_x[0]  This can be used if the initial state is known

    # kalman filter on z:
    z = tilde_z + torch.matmul(H_transpose, u_N)
    tilde_x_est_f, _ = pytorchEstimator(z, filterStateInit)
    tilde_e_k_given_k_minus_1 = tilde_x - tilde_x_est_f
    filteringErrorMeanEnergyPerBatch = calcTimeSeriesMeanEnergy(tilde_e_k_given_k_minus_1)

    loss = - filteringErrorMeanEnergyPerBatch.mean()  # mean error energy of a single batch [W]

    scheduler.step(loss)
    loss.backward()
    optimizer.step()  # parameter update

    # kalman filter on tilde_z for printing relative player contribution:
    pure_tilde_x_est_f, _ = pytorchEstimator(tilde_z, filterStateInit)
    pure_tilde_e_k_given_k_minus_1 = tilde_x - pure_tilde_x_est_f
    pure_filteringErrorMeanEnergyPerBatch = calcTimeSeriesMeanEnergy(pure_tilde_e_k_given_k_minus_1)
    pureLoss = - pure_filteringErrorMeanEnergyPerBatch.mean()

    performance_vs_noPlayer = watt2dbm(-loss.item()) - watt2dbm(-pureLoss.item())  # db

    if performance_vs_noPlayer > bestPerformanceVsNoPlayer + saveThr:
        bestPerformanceVsNoPlayer = performance_vs_noPlayer
        bestgap = theoreticalPlayerImprovement[p-1] - bestPerformanceVsNoPlayer
        torch.save(player.state_dict(), playerFileName)
        print('player saved')


    print(f'epoch {epoch}: MSE w.r.t pure input is {performance_vs_noPlayer} db; gap from theoretical: {theoreticalPlayerImprovement[p-1] - performance_vs_noPlayer}; lr: {scheduler._last_lr[-1]}')

    if scheduler._last_lr[-1] < lowThrLr:
        print(f'Stoping optimization due to learning rate of {scheduler._last_lr[-1]}')
        print(playerType + f': Best gap performance vs theoretical is: {bestgap}')
        break



