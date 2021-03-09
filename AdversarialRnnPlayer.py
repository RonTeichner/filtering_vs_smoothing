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
useCuda = False
if useCuda:
    device = 'cuda'
else:
    device = 'cpu'

batchSize = 20
dp = False

playerType = 'NoAccess'  # {'NoAccess', 'Causal', 'Genie'}

savedList = pickle.load(open(fileName + '_final_' + '.pt', "rb"))
sysModel, bounds_N, currentFileName_N, bounds_N_plus_m, currentFileName_N_plus_m, bounds_N_plus_2m, currentFileName_N_plus_2m, mistakeBound, delta_trS, gapFromInfBound = savedList
savedList = pickle.load(open(currentFileName_N_plus_2m, "rb"))
N = savedList[2]

dim_x, dim_z = sysModel['F'].shape[0], sysModel['H'].shape[1]
Q_cholesky, R_cholesky = torch.tensor(np.linalg.cholesky(sysModel['Q']), dtype=torch.float, device=device), torch.tensor(np.linalg.cholesky(sysModel['R']), dtype=torch.float, device=device)

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=False, useCuda=useCuda)

# class definition
class GRU_Adversarial(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRU_Adversarial, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup RNN layer
        self.Adv_rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def normalizeControlEnergy(self, u_NxN):
        N, batchSize, useCuda = u_NxN.shape[0], u_NxN.shape[1], u_NxN.is_cuda
        u_NxN = u_NxN.reshape(N, batchSize, N, -1)  # shape: [N, batchSize, N, dim_x]
        dim_x = u_NxN.shape[3]

        if useCuda:
            totalLeftEnergy = N*torch.ones(batchSize, dtype=torch.float).cuda()
            u_N = torch.zeros(N, batchSize, dim_x, dtype=torch.float).cuda()
            N_tensor = torch.tensor(torch.sqrt(N), dtype=torch.float).cuda()
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

    def forward(self, processNoiseBlockVec, measurementNoiseBlockVec, hidden):
        # processNoiseBlockVec, measurementNoiseBlockVec shapes: [N, batchSize, dim_x*N]
        controlHiddenDim, hidden = self.Adv_rnn(torch.cat((processNoiseBlockVec, measurementNoiseBlockVec), dim=2), hidden)
        # controlHiddenDim shape: [N, batchSize, hidden_dim]
        control_dim_x = self.linear(controlHiddenDim)
        # control_dim_x shape: [N, batchSize, dim_x*N]
        control_dim_x = self.normalizeControlEnergy(control_dim_x)

        return control_dim_x[:, :, :, None], hidden

# Define player
print("Build RNN player model ...")
num_layers, hidden_size = 1, 60
player = GRU_Adversarial(input_dim=N*(dim_x + dim_z), hidden_dim=hidden_size, output_dim=N*dim_x, num_layers=num_layers).to(device=device)

optimizer = optim.Adam(player.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, threshold=1e-6)

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
    u_N, _ = player(processNoisesKnown2Player[:, :, :, :, 0].reshape(N, batchSize, -1), measurementNoisesKnown2Player[:, :, :, :, 0].reshape(N, batchSize, -1), hidden)

    #  u_N.pow(2).sum(dim=2).sum(dim=0)  # this shows the energy used by the player at every batch

    # estimator init values:
    filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(batchSize, dim_x, 1))
    if dp: print(f'filter init mean error energy w.r.t trace(bar(sigma)): {watt2dbm(np.mean(np.power(np.linalg.norm(filterStateInit, axis=1), 2), axis=0)) - watt2dbm(np.trace(filter_P_init))} db')
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
    # filterStateInit = tilde_x[0]  This can be used if the initial state is known

    # kalman filter:
    z = tilde_z + torch.matmul(H_transpose, u_N)
    tilde_x_est_f, _ = pytorchEstimator(z, filterStateInit)
    tilde_e_k_given_k_minus_1 = tilde_x - tilde_x_est_f
    filteringErrorMeanEnergyPerBatch = calcTimeSeriesMeanEnergy(tilde_e_k_given_k_minus_1)

    loss = - filteringErrorMeanEnergyPerBatch.mean()  # mean error energy of a single batch [W]

    #  scheduler.step(energyEfficience.max())
    loss.backward()
    optimizer.step()  # parameter update



