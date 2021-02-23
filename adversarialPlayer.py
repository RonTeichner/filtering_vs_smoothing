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

fileName = 'sys2D.pt'
enableSym = False
if enableSym:
    np.random.seed(13)  #  for 2D systems, seed=13 gives two control angles, seed=10 gives multiple angles, seed=9 gives a single angle

    dim_x, dim_z = 2, 2
    N = 20  # time steps
    batchSize = 1
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

    # Smart players:
    if enableSmartPlayers:
        u_1, u_2, u_3 = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float), torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float), torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
        if useCuda:
            u_1, u_2, u_3 = u_1.cuda(), u_2.cuda(), u_3.cuda()
        u_1, _ = noAccessPlayer(adversarialPlayersToolbox, u_1, torch.zeros_like(tilde_e_k_given_k_minus_1)) # tilde_e_k_given_k_minus_1 is given only for the window size calculation. It is legit
        u_3 = geniePlayer(adversarialPlayersToolbox, u_3, tilde_e_k_given_k_minus_1)
        u_2 = causalPlayer(adversarialPlayersToolbox, u_2, processNoises, tilde_x[0:1])

        enableTestEnergyFactor = True
        if enableTestEnergyFactor:
            u_3_doubleEnergy = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float)
            u_3_doubleEnergy = geniePlayer(adversarialPlayersToolbox, u_3_doubleEnergy, tilde_e_k_given_k_minus_1, 2)
            print(f'mean energy of u_3: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_3).mean())}, ' [dbm]')
            print(f'mean energy of u_3_doubleEnergy: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_3_doubleEnergy).mean())}, ' [dbm]')
            plt.figure()
            batchIdx = 0
            plt.plot(volt2dbm(np.linalg.norm(u_3[:, batchIdx:batchIdx+1].cpu().numpy(), axis=2))[:, 0, 0] - volt2dbm(np.linalg.norm(u_3_doubleEnergy[:, batchIdx:batchIdx+1].cpu().numpy(), axis=2))[:, 0, 0], label=r'$\frac{||u_N(N)^{(3)}||_2}{||u_N(2N)^{(3)}||_2}$')
            plt.ylabel('db')
            plt.xlabel('k')
            plt.grid()
            plt.legend()
            #plt.show()

        print(f'mean energy of u_1: ',{watt2dbm(calcTimeSeriesMeanEnergy(u_1).mean())},' [dbm]')
        print(f'mean energy of u_2: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_1).mean())}, ' [dbm]')
        print(f'mean energy of u_3: ', {watt2dbm(calcTimeSeriesMeanEnergy(u_3).mean())}, ' [dbm]')

        z_1 = tilde_z + torch.matmul(H_transpose, u_1)
        x_1_est_f, x_1_est_s = pytorchEstimator(z_1, filterStateInit)

        z_2 = tilde_z + torch.matmul(H_transpose, u_2)
        x_2_est_f, x_2_est_s = pytorchEstimator(z_2, filterStateInit)

        z_3 = tilde_z + torch.matmul(H_transpose, u_3)
        x_3_est_f, x_3_est_s = pytorchEstimator(z_3, filterStateInit)


    pickle.dump([sysModel, tilde_z, tilde_x, processNoises, measurementNoises, filter_P_init, filterStateInit, u_0, u_1, u_2, u_3, tilde_x_est_f, x_0_est_f, x_1_est_f, x_2_est_f, x_3_est_f,
                 pytorchEstimator.theoreticalBarSigma, pytorchEstimator.normalizedNoKnowledgePlayerContribution, adversarialPlayersToolbox.theoretical_lambda_Xi_N_max, adversarialPlayersToolbox.lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max], open(fileName, 'wb'))

# plotting:
adversarialPlayerPlotting(fileName)

