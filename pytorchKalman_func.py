import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
from filterpy.common import Q_discrete_white_noise, kinematic_kf, Saver
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from __future__ import print_function
#from __future__ import division
from analyticResults_func import watt2db, volt2dbm, watt2dbm, calc_tildeB, calc_tildeC


def GenSysModel(dim_x, dim_z):
    if dim_x == 1:
        F = -1 + 2 * np.random.rand(dim_x, dim_x)
    else:
        F = np.random.randn(dim_x, dim_x)
        eigAbsMax = np.abs(np.linalg.eigvals(F)).max()
        F = F/((1.1+0.1*np.random.rand(1))*eigAbsMax)

    H = np.random.randn(dim_x, dim_z)
    H = H/np.linalg.norm(H)

    processNoiseVar, measurementNoiseVar = 1/dim_x, 1/dim_x
    Q = processNoiseVar * np.eye(dim_x)
    R = measurementNoiseVar * np.eye(dim_z)
    return {"F": F, "H": H, "Q": Q, "R": R}

def GenMeasurements(N, batchSize, sysModel):
    F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]
    dim_x, dim_z = F.shape[0], H.shape[1]
    # generate state
    x, z = np.zeros((N, batchSize, dim_x, 1)), np.zeros((N, batchSize, dim_z, 1))
    P = np.eye(dim_x)
    x[0] = np.matmul(np.linalg.cholesky(P), np.random.randn(batchSize, dim_x, 1))

    processNoises = np.matmul(np.linalg.cholesky(Q), np.random.randn(N, batchSize, dim_x, 1))
    measurementNoises = np.matmul(np.linalg.cholesky(R), np.random.randn(N, batchSize, dim_z, 1))

    print(f'amount of energy into the system is {watt2dbm(np.sum(np.power(np.linalg.norm(processNoises[:,0]), 2)))} dbm')

    for i in range(1, N):
        x[i] = np.matmul(F, x[i - 1]) + processNoises[i - 1]

    print(f'amount of energy out from the system is {watt2dbm(np.sum(np.power(np.linalg.norm(x[:,0]), 2)))} dbm')

    z = np.matmul(H.transpose(), x) + measurementNoises

    return z, x

def Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit):
    # filter_P_init: [1, batchSize, dim_x, dim_x]
    # filterStateInit: [1, batchSize, dim_x, 1]
    # z: [N, batchSize, dim_z, 1]
    F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]
    dim_x, dim_z = F.shape[0], H.shape[1]
    N, batchSize = z.shape[0], z.shape[1]

    # define estimator
    k_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    k_filter.Q, k_filter.R, k_filter.H, k_filter.F = Q, R, H.transpose(), F

    # run estimator on measurements:
    x_est_f, x_est_s = np.zeros((N, batchSize, dim_x, 1)), np.zeros((N, batchSize, dim_x, 1))
    for b in range(batchSize):
        k_filter.x, k_filter.P = filterStateInit[0, b].copy(), filter_P_init[0, b].copy()
        x_est, cov, x_est_f_singleTimeSeries, _ = k_filter.batch_filter(zs=z[:, b], update_first=False)
        x_est_s_singleTimeSeries, _, _, _ = k_filter.rts_smoother(x_est, cov)
        x_est_f[:, b], x_est_s[:, b] = x_est_f_singleTimeSeries, x_est_s_singleTimeSeries

    return x_est_f, x_est_s

def Pytorch_filter_smoother(z, sysModel, filterStateInit):
    # filter_P_init: [1, batchSize, dim_x, dim_x] is not in use because this filter works from the start on the steady-state-gain
    # filterStateInit: [1, batchSize, dim_x, 1]
    # z: [N, batchSize, dim_z, 1]
    F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]

    theoreticalBarSigma = solve_discrete_are(a=np.transpose(F), b=H, q=Q, r=R)
    Ka_0 = np.dot(theoreticalBarSigma, np.dot(H, np.linalg.inv(np.dot(np.transpose(H), np.dot(theoreticalBarSigma, H)) + R)))  # first smoothing gain
    K = np.dot(F, Ka_0) # steadyKalmanGain
    tildeF = F - np.dot(K, np.transpose(H))
    Sint = np.matmul(np.linalg.inv(np.matmul(F, theoreticalBarSigma)), K)
    thr = 1e-20 * np.abs(tildeF).max()

    # stuff to cuda:
    tildeF = torch.tensor(tildeF, dtype=torch.float).cuda()
    tildeF_transpose = torch.transpose(tildeF, 1, 0)
    K = torch.tensor(K, dtype=torch.float).cuda()
    z = torch.tensor(z, dtype=torch.float).cuda()
    H = torch.tensor(H, dtype=torch.float).cuda()
    Sint = torch.tensor(Sint, dtype=torch.float).cuda()
    H_transpose = torch.transpose(H, 1, 0)
    thr = torch.tensor(thr, dtype=torch.float).cuda()
    theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float).cuda()

    # filtering, inovations:
    dim_x, dim_z = F.shape[0], H.shape[1]
    N, batchSize = z.shape[0], z.shape[1]
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float).cuda()
    hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float).cuda()  # hat_x_k_plus_1_given_k is in index [k+1]
    bar_z_k = torch.zeros(N, batchSize, dim_z, 1, dtype=torch.float).cuda()
    hat_x_k_plus_1_given_k[0] = torch.matmul(tildeF, filterStateInit[0]) + torch.matmul(K, z[0])
    bar_z_k[0] = z[0]
    for k in range(N-1):
        hat_x_k_plus_1_given_k[k+1] = torch.matmul(tildeF, hat_x_k_plus_1_given_k[k]) + torch.matmul(K, z[k])
    for k in range(N):
        bar_z_k[k] = z[k] - torch.matmul(H_transpose, hat_x_k_plus_1_given_k[k])

    # smoothing:
    hat_x_k_given_N = torch.zeros(N, batchSize, dim_x, 1, dtype=torch.float).cuda()
    for k in range(N):
        # for i==k:
        Ka_i_minus_k = torch.matmul(theoreticalBarSigma, Sint)
        hat_x_k_given_i = hat_x_k_plus_1_given_k[k] + torch.matmul(Ka_i_minus_k, bar_z_k[k])
        for i in range(k+1, N):
            Ka_i_minus_k = torch.matmul(theoreticalBarSigma, torch.matmul(torch.matrix_power(tildeF_transpose, i-k), Sint))
            hat_x_k_given_i = hat_x_k_given_i + torch.matmul(Ka_i_minus_k, bar_z_k[i])

            if torch.max(torch.abs(Ka_i_minus_k)) < thr:
                break
        hat_x_k_given_N[k] = hat_x_k_given_i

    #  x_est_f, x_est_s =  hat_x_k_plus_1_given_k, hat_x_k_given_N - these are identical values
    return hat_x_k_plus_1_given_k, hat_x_k_given_N

# class definition
class Pytorch_filter_smoother_Obj(nn.Module):
    def __init__(self, sysModel, enableSmoothing = True, useCuda=True):
        super(Pytorch_filter_smoother_Obj, self).__init__()
        self.useCuda = useCuda
        self.enableSmoothing = enableSmoothing
        # filter_P_init: [1, batchSize, dim_x, dim_x] is not in use because this filter works from the start on the steady-state-gain
        # filterStateInit: [1, batchSize, dim_x, 1]
        # z: [N, batchSize, dim_z, 1]
        F, H, Q, R = sysModel["F"], sysModel["H"], sysModel["Q"], sysModel["R"]

        self.dim_x, self.dim_z = F.shape[0], H.shape[1]

        theoreticalBarSigma = solve_discrete_are(a=np.transpose(F), b=H, q=Q, r=R)
        Ka_0 = np.dot(theoreticalBarSigma, np.dot(H, np.linalg.inv(np.dot(np.transpose(H), np.dot(theoreticalBarSigma, H)) + R)))  # first smoothing gain
        K = np.dot(F, Ka_0)  # steadyKalmanGain
        tildeF = F - np.dot(K, np.transpose(H))
        Sint = np.matmul(np.linalg.inv(np.matmul(F, theoreticalBarSigma)), K)
        thr = 1e-20 * np.abs(tildeF).max()

        DeltaFirstSample = np.dot(Ka_0, np.dot(np.transpose(H), theoreticalBarSigma))
        theoreticalSmoothingFilteringDiff = solve_discrete_lyapunov(a=np.dot(theoreticalBarSigma, np.dot(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma))) , q=DeltaFirstSample)
        theoreticalSmoothingSigma = theoreticalBarSigma - theoreticalSmoothingFilteringDiff

        A_N = solve_discrete_lyapunov(a=tildeF, q=np.eye(self.dim_x))
        normalizedNoKnowledgePlayerContribution = np.trace(np.matmul(np.dot(H, np.transpose(K)), np.matmul(np.transpose(A_N), np.dot(K, np.transpose(H)))))

        smootherRecursiveGain = np.matmul(theoreticalBarSigma, np.matmul(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma)))
        smootherGain = np.linalg.inv(F) - smootherRecursiveGain

        '''
        print(f'The eigenvalues of tildeF: {np.linalg.eig(tildeF)[0]}')
        print(f'The eigenvalues of KH\': {np.linalg.eig(np.matmul(K, np.transpose(H)))[0]}')
        print(f'The eigenvalues of smootherRecursiveGain: {np.linalg.eig(smootherRecursiveGain)[0]}')
        print(f'The eigenvalues of smootherGain: {np.linalg.eig(smootherGain)[0]}')
        '''
        # stuff to cuda:
        if self.useCuda:
            self.tildeF = torch.tensor(tildeF, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.tildeF_transpose = torch.tensor(tildeF.transpose(), dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.K = torch.tensor(K, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.H = torch.tensor(H, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.Sint = torch.tensor(Sint, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.H_transpose = torch.tensor(H.transpose(), dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.thr = torch.tensor(thr, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.theoreticalSmoothingSigma = torch.tensor(theoreticalSmoothingSigma, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.normalizedNoKnowledgePlayerContribution = torch.tensor(normalizedNoKnowledgePlayerContribution, dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.tildeF = torch.tensor(tildeF, dtype=torch.float, requires_grad=False).contiguous()
            self.tildeF_transpose = torch.tensor(tildeF.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.K = torch.tensor(K, dtype=torch.float, requires_grad=False).contiguous()
            self.H = torch.tensor(H, dtype=torch.float, requires_grad=False).contiguous()
            self.Sint = torch.tensor(Sint, dtype=torch.float, requires_grad=False).contiguous()
            self.H_transpose = torch.tensor(H.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.thr = torch.tensor(thr, dtype=torch.float, requires_grad=False).contiguous()
            self.theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float, requires_grad=False).contiguous()
            self.theoreticalSmoothingSigma = torch.tensor(theoreticalSmoothingSigma, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous()
            self.normalizedNoKnowledgePlayerContribution = torch.tensor(normalizedNoKnowledgePlayerContribution, dtype=torch.float, requires_grad=False).contiguous()

    def forward(self, z, filterStateInit):
        # z, filterStateInit are cuda

        # filtering
        N, batchSize = z.shape[0], z.shape[1]

        if self.useCuda:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False).cuda()  #  hat_x_k_plus_1_given_k is in index [k+1]            
        else:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False)  #  hat_x_k_plus_1_given_k is in index [k+1]            

        hat_x_k_plus_1_given_k[0] = filterStateInit

        hat_x_k_plus_1_given_k[1] = torch.matmul(self.tildeF, hat_x_k_plus_1_given_k[0]) + torch.matmul(self.K, z[0])
        K_dot_z = torch.matmul(self.K, z)
        for k in range(N - 1):
            hat_x_k_plus_1_given_k[k + 1] = torch.matmul(self.tildeF, hat_x_k_plus_1_given_k[k].clone()) + K_dot_z[k]

        # smoothing:
        if self.useCuda:
            hat_x_k_given_N = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False).cuda()
        else:
            hat_x_k_given_N = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False)

        if self.enableSmoothing:
            bar_z_N_minus_1 = z[N - 1] - torch.matmul(self.H_transpose, hat_x_k_plus_1_given_k[N - 1]) # smoother init val
            hat_x_k_given_N[N-1] = hat_x_k_plus_1_given_k[N-1] + torch.matmul(self.Ka_0, bar_z_N_minus_1)
            filteringInput = torch.matmul(self.smootherGain, hat_x_k_plus_1_given_k)
            for k in range(N-2, -1, -1):
                hat_x_k_given_N[k] = torch.matmul(self.smootherRecursiveGain, hat_x_k_given_N[k+1].clone()) + filteringInput[k+1]#torch.matmul(self.smootherGain, hat_x_k_plus_1_given_k[k+1])

        #  x_est_f, x_est_s =  hat_x_k_plus_1_given_k, hat_x_k_given_N - these are identical values

        return hat_x_k_plus_1_given_k, hat_x_k_given_N

def constantMaximizeFilteringInputSearch(sysModel, N):
    dim_x = sysModel["F"].shape[0]
    assert dim_x == 2
    batchSize = 1
    uAngles = np.linspace(-np.pi, np.pi, 360*10)
    filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
    objectivePowerEfficiency = np.zeros(uAngles.shape[0])
    for i,uAngle in enumerate(uAngles):
        print(f'constantMaximizeFilteringInputSearch: {i/uAngles.shape[0]*100} %')
        u = np.concatenate((np.cos(uAngle)*np.ones((N, 1, 1, 1)), np.sin(uAngle)*np.ones((N, 1, 1, 1))), axis=2)
        z = np.matmul(np.transpose(sysModel["H"]), u)
        x_est_f, x_est_s = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
        objectiveMeanPowerPerBatch = np.power(x_est_f[1:], 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u[:-1], 2).sum(axis=2).mean(axis=0)
        objectivePowerEfficiency[i] = np.divide(objectiveMeanPowerPerBatch, inputMeanPowerPerBatch)

    optimalAngle = uAngles[np.argmax(objectivePowerEfficiency)]
    uOptimal = np.concatenate((np.cos(optimalAngle)*np.ones((N, 1, 1, 1)), np.sin(optimalAngle)*np.ones((N, 1, 1, 1))), axis=2)

    plt.figure()
    plt.plot(uAngles/np.pi*180, watt2db(objectivePowerEfficiency), label=r'$\frac{\sum_{k=1}^{N-1} ||\xi_k||_2^2}{\sum_{k=0}^{N-2} ||x^u_k||_2^2}$')
    plt.xlabel('deg')
    plt.title('Filtering energy efficiency vs unit-input direction')
    plt.ylabel('db')
    plt.grid()
    plt.legend()

    return uOptimal

def smoothingOptimalInitCalc(sysModel):
    N = 100000
    dim_x = sysModel["F"].shape[0]
    theoreticalBarSigma = solve_discrete_are(a=np.transpose(sysModel["F"]), b=sysModel["H"], q=sysModel["Q"], r=sysModel["R"])
    Ka_0 = np.dot(theoreticalBarSigma, np.dot(sysModel["H"], np.linalg.inv(np.dot(np.transpose(sysModel["H"]), np.dot(theoreticalBarSigma, sysModel["H"])) + sysModel["R"])))  # first smoothing gain
    K = np.dot(sysModel["F"], Ka_0)  # steadyKalmanGain
    tildeF = sysModel["F"] - np.dot(K, np.transpose(sysModel["H"]))
    inv_F_Sigma = np.linalg.inv(np.matmul(sysModel["F"], theoreticalBarSigma))
    K_HT = np.matmul(K, sysModel["H"].transpose())
    D_int = np.matmul(inv_F_Sigma, K_HT)
    smootherRecursiveGain = np.matmul(theoreticalBarSigma, np.matmul(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma)))
    tildeB_1_0 = calc_tildeB(tildeF, theoreticalBarSigma, D_int, 1, 0, N)
    thr = 1e-20 * np.abs(tildeF).max()

    # calc past:
    tildeB_1_i = tildeB_1_0 # i=k-1=0
    past = tildeB_1_0
    i=0
    while True:
        i -= 1
        tildeB_1_i = np.matmul(tildeB_1_i, tildeF)
        past = past + tildeB_1_i
        if np.abs(tildeB_1_i).max() < thr:
            break

    # calc future:
    tildeC_k_k = calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, 1, 1, N)  # i=k=1
    tildeC_k_i = tildeC_k_k  # i=k=1
    future = tildeC_k_i
    i=1
    while True:
        i += 1
        tildeC_k_i = np.matmul(smootherRecursiveGain, tildeC_k_i)
        future = future + tildeC_k_i
        if np.abs(tildeC_k_i).max() < thr:
            break

    A = np.matmul(past + future, K_HT)
    w, v = np.linalg.eig(np.matmul(np.transpose(A), A))
    x_u_star = v[:, np.argmax(np.abs(w)):np.argmax(np.abs(w))+1]

    filterStateInit = np.matmul(np.matmul(np.linalg.inv(np.eye(dim_x) - tildeF), K_HT), x_u_star)

    return filterStateInit, x_u_star

def calcTimeSeriesMeanRootEnergy(x):
    return torch.mean(torch.norm(x, dim=2), dim=0)

def calcTimeSeriesMeanEnergy(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=2), dim=0)

def calcTimeSeriesMeanEnergyRunningAvg(x):
    square_norm_x = torch.sum(torch.pow(x, 2), dim=2)
    cumsum_square_norm_x = torch.cumsum(square_norm_x, dim=0)
    return torch.div(cumsum_square_norm_x, torch.cumsum(torch.ones_like(cumsum_square_norm_x), dim=0))

def noKnowledgePlayer(u):
    dim_x = u.shape[2]
    sigma_u_square = torch.tensor(1 / dim_x, dtype=torch.float)
    return torch.mul(torch.sqrt(sigma_u_square), torch.randn_like(u))

def noAccessPlayer(adversarialPlayersToolbox, u, tilde_e_k_given_k_minus_1):
    # tilde_e_k_given_k_minus_1 should be used only for the window size calculation
    use_cuda = adversarialPlayersToolbox.use_cuda
    N, batchSize, dim_x = u.shape[0], u.shape[1], u.shape[2]

    if use_cuda:
        caligraphE_Ni = torch.zeros(batchSize, N, 1, dtype=torch.float).cuda()
    else:
        caligraphE_Ni = torch.zeros(batchSize, N, 1, dtype=torch.float)

    enableCalculateForAllWindows = False
    if enableCalculateForAllWindows:
        startWindowLength = 1
    else:
        startWindowLength = N

    for Ni in range(startWindowLength,N+1):
        print(f'adversarial no access player {Ni} out of {N}')
        blockVec_tilde_e_full = tilde_e_k_given_k_minus_1[:Ni].permute(1, 0, 2, 3).reshape(batchSize, Ni * dim_x, 1)

        Xi_Ni = adversarialPlayersToolbox.compute_Xi_l_N(0, Ni)
        # note that J_0_N = Xi_N
        J_0_Ni = (Xi_Ni, 0, Ni)
        if use_cuda:
            tilde_b = torch.zeros(batchSize, Ni, 1, dtype=torch.float).cuda()
            alpha = Ni * torch.ones(batchSize, dtype=torch.float).cuda()
        else:
            tilde_b = torch.zeros(batchSize, Ni, 1, dtype=torch.float)
            alpha = Ni * torch.ones(batchSize, dtype=torch.float)

        u_Ni_blockVec, _ = adversarialPlayersToolbox.corollary_4_opt(J_j_N=J_0_Ni, tilde_b=tilde_b, alpha=alpha)
        caligraphE_Ni[:, Ni-1:Ni] = adversarialPlayersToolbox.compute_caligraphE(u_Ni_blockVec, blockVec_tilde_e_full)

    u[:N] = u_Ni_blockVec.reshape(batchSize, N, dim_x, 1).permute(1, 0, 2, 3)
    if enableCalculateForAllWindows:
        plt.figure()
        plt.plot(np.arange(1, N+1), watt2dbm(caligraphE_Ni.cpu().numpy()[0, :, 0]), label=r'empirical ${\cal E}^{(1)}_{F,N}$')
        plt.xlabel('N')
        plt.ylabel('dbm')
        plt.legend()
        plt.grid()

    return u, caligraphE_Ni

class playersToolbox:
    def __init__(self, pytorchEstimator, delta_u, delta_caligraphE, Ns_2_2N0_factor):
        self.tildeF = pytorchEstimator.tildeF
        self.K = pytorchEstimator.K
        self.H = pytorchEstimator.H
        self.f = pytorchEstimator
        self.dim_x = self.tildeF.shape[0]
        self.delta_u, self.delta_caligraphE, self.Ns_2_2N0_factor = delta_u, delta_caligraphE, Ns_2_2N0_factor
        self.use_cuda = self.tildeF.is_cuda
        self.thr_db = 0.01  # db
        self.thr = 1e-20 * torch.max(torch.abs(self.tildeF))

        self.K_HT = torch.matmul(self.K, torch.transpose(self.H, 1, 0))

        self.summed = torch.zeros(self.dim_x, self.dim_x, dtype=torch.float)

        self.compute_Xi_l_N_previous_l, self.compute_Xi_l_N_previous_N = 0, 0
        self.compute_bar_Xi_N_previous_N = 0
        self.compute_J_j_N_previous_j, self.compute_J_j_N_previous_N = 0, 0
        self.compute_J_j_N_eig_previous_j, self.compute_J_j_N_eig_previous_N = 0, 0
        self.compute_tildeJ_N0_j_N_previous_N_0, self.compute_tildeJ_N0_j_N_previous_j, self.compute_tildeJ_N0_j_N_previous_N = 0, 0, 0

        if self.use_cuda:
            self.K_HT.cuda()
            self.summed.cuda()
            self.thr_db.cuda()
            self.thr.cuda()

        self.compute_lambda_Xi_max(enableFigure=True)

    def compute_Xi_l_N(self, l, N):
        if not(l == self.compute_Xi_l_N_previous_l and N == self.compute_Xi_l_N_previous_N):
            self.compute_Xi_l_N_previous_l, self.compute_Xi_l_N_previous_N = l, N

            self.Xi_N_l = torch.zeros(N * self.dim_x, N * self.dim_x, dtype=torch.float)
            if self.use_cuda: self.Xi_N_l.cuda()

            for r in range(N):
                for c in range(N):
                    k = np.max((r, c, l - 1))
                    self.summed.fill_(0)
                    if self.use_cuda:
                        summedItter = torch.tensor(float("inf")).cuda()
                    else:
                        summedItter = torch.tensor(float("inf"))
                    while True:
                        k += 1
                        if torch.max(torch.abs(summedItter)) < self.thr or k > N-1:
                            break
                        summedItter = torch.matmul(torch.transpose(torch.matrix_power(self.tildeF, k-1-r), 1, 0), torch.matrix_power(self.tildeF, k-1-c))
                        self.summed = self.summed + summedItter

                    self.Xi_N_l[self.dim_x*r:self.dim_x*(r+1), self.dim_x*c:self.dim_x*(c+1)] = torch.matmul(torch.transpose(self.K_HT, 1, 0), torch.matmul(self.summed, self.K_HT))

        return self.Xi_N_l

    def compute_bar_Xi_N(self, N):
        if not(N == self.compute_bar_Xi_N_previous_N):
            self.compute_bar_Xi_N_previous_N = N
            self.bar_Xi_N = torch.zeros(N * self.dim_x, N * self.dim_x, dtype=torch.float)
            if self.use_cuda: self.bar_Xi_N.cuda()

            for r in range(N):
                for c in range(r):
                    self.bar_Xi_N[self.dim_x * r:self.dim_x * (r + 1), self.dim_x * c:self.dim_x * (c + 1)] = torch.matmul(torch.matrix_power(self.tildeF, r - 1 - c), self.K_HT)

        return self.bar_Xi_N

    def compute_J_j_N(self, j, N):
        if not(j == self.compute_J_j_N_previous_j and N == self.compute_J_j_N_previous_N):
            self.compute_J_j_N_previous_j, self.compute_J_j_N_previous_N = j, N
            self.J_j_N = torch.zeros((N-j)*self.dim_x, (N-j)*self.dim_x, dtype=torch.float)
            if self.use_cuda: self.J_j_N.cuda()

            for r in range(N-j):
                for c in range(N-j):
                    k = j + np.max((r, c))
                    self.summed.fill_(0)
                    if self.use_cuda:
                        summedItter = torch.tensor(float("inf")).cuda()
                    else:
                        summedItter = torch.tensor(float("inf"))
                    while True:
                        k += 1
                        if torch.max(torch.abs(summedItter)) < self.thr or k > N-1:
                            break
                        summedItter = torch.matmul(torch.transpose(torch.matrix_power(self.tildeF, k-1-r-j), 1, 0), torch.matrix_power(self.tildeF, k-1-c-j))
                        self.summed = self.summed + summedItter

                    self.J_j_N[self.dim_x*r:self.dim_x*(r+1), self.dim_x*c:self.dim_x*(c+1)] = torch.matmul(torch.transpose(self.K_HT, 1, 0), torch.matmul(self.summed, self.K_HT))

        return self.J_j_N

    def compute_tildeJ_N0_j_N(self, N_0, j, N):
        if not(N_0 == self.compute_tildeJ_N0_j_N_previous_N_0 and j == self.compute_tildeJ_N0_j_N_previous_j and N == self.compute_tildeJ_N0_j_N_previous_N):
            self.compute_tildeJ_N0_j_N_previous_N_0, self.compute_tildeJ_N0_j_N_previous_j, self.compute_tildeJ_N0_j_N_previous_N = N_0, j, N

            self.tildeJ_N0_j_N = torch.zeros(N_0*self.dim_x, (N-j)*self.dim_x, dtype=torch.float)
            if self.use_cuda: self.tildeJ_N0_j_N.cuda()

            for r in range(N_0):
                for c in range(N-j):
                    k = np.max((r-N_0+j, c+j))
                    self.summed.fill_(0)
                    if self.use_cuda:
                        summedItter = torch.tensor(float("inf")).cuda()
                    else:
                        summedItter = torch.tensor(float("inf"))
                    while True:
                        k += 1
                        if torch.max(torch.abs(summedItter)) < self.thr or k > N-1:
                            break
                        summedItter = torch.matmul(torch.transpose(torch.matrix_power(self.tildeF, k-1-r-j+N_0), 1, 0), torch.matrix_power(self.tildeF, k-1-c-j))
                        self.summed = self.summed + summedItter

                    self.tildeJ_N0_j_N[self.dim_x*r:self.dim_x*(r+1), self.dim_x*c:self.dim_x*(c+1)] = torch.matmul(torch.transpose(self.K_HT, 1, 0), torch.matmul(self.summed, self.K_HT))

        return self.tildeJ_N0_j_N

    def compute_tildeb_1(self, blockVec_u, N_0, j, N):
        if blockVec_u.shape[1] == 0:
            tildeb_1 = torch.zeros_like(blockVec_u)
        else:
            tildeJ_N0_j_N = self.compute_tildeJ_N0_j_N(N_0, j, N)
            tildeb_1 = torch.matmul(torch.transpose(tildeJ_N0_j_N, 1, 0), blockVec_u)
        return tildeb_1

    def compute_lambda_Xi_max(self, enableFigure):
        lambda_Xi_max, corresponding_eigenvector = list(), list()
        u_0 = np.zeros((1000, self.dim_x, 1))
        N=0
        while True:
            N += 1
            e, v = np.linalg.eig(self.compute_Xi_l_N(0, N).cpu().numpy())
            maxEigenvalueIndex = np.argmax(np.real(e))
            lambda_Xi_max.append(np.real(e[maxEigenvalueIndex]))
            corresponding_eigenvector.append(np.sqrt(N)*np.real(v[:, maxEigenvalueIndex]))
            u_0[N-1] = corresponding_eigenvector[N-1][:self.dim_x][:, None]
            if N > 20:
                stepsDiff = 1  #  int(N/2)
                #lambda_diff_db = np.abs(watt2dbm(lambda_Xi_max[-1]) - watt2dbm(lambda_Xi_max[-1-stepsDiff]))
                #delta_u = u_0[N-1] - u_0[N-1-stepsDiff]
                #u_0_diff_db = np.abs(volt2dbm(np.linalg.norm(u_0[N-1])) - volt2dbm(np.linalg.norm(u_0[N-1-stepsDiff])))
                #print(f'N = {N}: lambda diff: {lambda_diff_db} db, u_0 diff: {u_0_diff_db} db')
                print(f'{N}')
                if N >= 100 or \
                        (N > 20 and self.windowSizeTest(torch.tensor(u_0[N-1-stepsDiff][None, None, :, :], dtype=torch.float), torch.tensor(u_0[N-1][None, None, :, :], dtype=torch.float),
                                                        torch.tensor(lambda_Xi_max[-1-stepsDiff], dtype=torch.float), torch.tensor(lambda_Xi_max[-1], dtype=torch.float)
                                                        , 1, 1)):
                    break
        if self.use_cuda:
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous()

        u_0 = u_0[:N]

        if enableFigure:
            lambda_Xi_max = np.asarray(lambda_Xi_max)
            lambda_Xi_max_relation = np.divide(lambda_Xi_max[1:], lambda_Xi_max[:-1])
            #u_0 = np.zeros((lambda_Xi_max.shape[0], self.dim_x, 1))
            #for i in range(len(corresponding_eigenvector)):
                #u_0[i] = corresponding_eigenvector[i][:self.dim_x][:, None]
                #print(f'square norm of eigenvector for {i+1} time steps: {np.power(np.linalg.norm(corresponding_eigenvector[i]), 2)}')
            eigenvectors_energy = np.sum(np.power(u_0, 2), axis=1)
            if self.dim_x == 2:
                eigenvectors_angle_deg = np.arctan2(u_0[:,1], u_0[:,0]) / np.pi * 180
            eigenvectors_energy_relation = np.divide(eigenvectors_energy[1:], eigenvectors_energy[:-1])
            #delta_eigenvectors_energy = np.sum(np.power(eigenvectors[1:] - eigenvectors[:-1], 2), axis=1)
            #delta_eigenvector_energy_relation = np.divide(delta_eigenvectors_energy, np.sum(np.power(eigenvectors[:-1], 2), axis=1))
            #delta_eigenvalue_energy = lambda_Xi_max[1:] / lambda_Xi_max[:-1]

            plt.figure(figsize=(12, 4))
            plt.subplot(2, 2, 1)
            plt.plot(np.arange(1, N+1), watt2dbm(lambda_Xi_max), label=r'$\lambda^{\Xi_{N}}_{max}$')
            plt.ylabel('dbm')
            plt.xlabel('N')
            plt.grid()
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(np.arange(1, N+1), watt2dbm(eigenvectors_energy), label=r'$||u_{N}[0]||_2^2$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
            plt.ylabel('dbm')
            plt.xlabel('N')
            plt.legend()
            #plt.title(r'$\lambda^{\Xi_N}_{max}$')
            plt.grid()

            plt.subplot(2, 2, 4)
            plt.plot(np.arange(2, N+1), watt2db(eigenvectors_energy_relation), label=r'$\frac{||u_{N+1}[0]||_2^2}{||u_{N}[0]||_2^2}$')
            plt.ylabel('db')
            plt.xlabel('N')
            plt.legend()
            #plt.title(r'$\lambda^{\Xi_N}_{max}$')
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.plot(np.arange(2, N+1), watt2db(lambda_Xi_max_relation), label=r'$\frac{\lambda^{\Xi_{N+1}}_{max}}{\lambda^{\Xi_{N}}_{max}}$')
            plt.ylabel('db')
            plt.xlabel('N')
            plt.legend()
            #plt.title(r'$\lambda^{\Xi_N}_{max}$')
            plt.grid()

            plt.subplots_adjust(hspace=0.4)
            plt.subplots_adjust(wspace=0.4)

            # plotting the time-series control for longest N available
            blockVec_u = corresponding_eigenvector[-1]
            u = blockVec_u.reshape(int(blockVec_u.shape[0] / self.dim_x), self.dim_x, 1)
            u_energy = np.power(np.linalg.norm(u, axis=1, keepdims=True), 2)
            plt.figure()
            if self.dim_x == 2: plt.subplot(2,1,1)
            plt.plot(np.arange(0, N), watt2dbm(u_energy)[:, 0, 0], label=r'$||u_k||_2^2$')
            plt.title(f'Energy of control for {N} time-steps horizon')
            plt.xlabel('k')
            plt.ylabel('dbm')
            plt.legend()
            plt.grid()

            if self.dim_x == 2:
                u_angle_deg = np.arctan2(u[:,1], u[:,0]) / np.pi * 180
                plt.subplot(2,1,2)
                plt.plot(np.arange(0, N), u_angle_deg[:, 0], '--*', label=r'$\angle u_k$')
                plt.title(f'Angle of control for {N} time-steps horizon')
                plt.xlabel('k')
                plt.ylabel('deg')
                plt.legend()
                plt.grid()
                plt.subplots_adjust(hspace=0.5)

            # plotting u[0] for all tested N
            if self.dim_x == 2:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(np.arange(1, N + 1), watt2dbm(eigenvectors_energy), '--*', label=r'$||u_{N}[0]||_2^2$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
                plt.ylabel('dbm')
                plt.xlabel('N')
                plt.legend()
                plt.grid()

                plt.subplot(2,1,2)
                plt.plot(np.arange(1, N + 1), eigenvectors_angle_deg, '--*', label=r'$\angle u_{N}[0]$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
                plt.ylabel('deg')
                plt.xlabel('N')
                plt.legend()
                plt.grid()

                plt.subplots_adjust(hspace=0.5)

            #plt.show()

    def windowSizeTest(self, u_N0_j, u_2N0_j, caligraphE_N0_j, caligraphE_2N0_j, alpha_over_N_0, alpha_over_2N_0):
        delta_u_energy = torch.sum(torch.pow(u_N0_j - u_2N0_j, 2), dim=2)
        u_N_0_energy, u_2N_0_energy = torch.sum(torch.pow(u_N0_j, 2), dim=2), torch.sum(torch.pow(u_2N0_j, 2), dim=2)
        u_delta_violation = torch.max(torch.div(delta_u_energy, u_2N_0_energy))
        u_N_0_violation, u_2N_0_violation = torch.max(torch.div(u_N_0_energy, alpha_over_N_0)), torch.max(torch.div(u_2N_0_energy, alpha_over_2N_0))
        caligraphE_violation = torch.max(torch.div(torch.abs(caligraphE_N0_j - caligraphE_2N0_j), torch.abs(caligraphE_2N0_j)))
        print(f'window size test: delta u violation: {u_delta_violation}; u_N_0 violation: {u_N_0_violation}; u_2N_0 violation: {u_2N_0_violation}, E violation: {caligraphE_violation}')
        #print(f'u_2N0_j/u_N0_j: {torch.div(u_2N0_j, u_N0_j)}')
        delta_u_condition = np.min((u_delta_violation.item(), np.max((u_N_0_violation.item(), u_2N_0_violation.item())))) < self.delta_u
        delta_caligraphE_condition = caligraphE_violation < self.delta_caligraphE
        #return delta_u_condition and delta_caligraphE_condition
        return delta_caligraphE_condition

    def noAccessPlayer_optParams(self, j, N_0, N, powerUsedSoFar, relevant_u):
        nSamples, batchSize, dim_x = relevant_u.shape[0], relevant_u.shape[1], relevant_u.shape[2]
        blockVec_u = relevant_u.permute(1, 0, 2, 3).reshape(batchSize, nSamples*dim_x, 1)
        if j < N_0:
            J_j_N = self.compute_J_j_N(j, j + N_0)
            J_j_N_param_j, J_j_N_param_N = j, j + N_0
            b = self.compute_tildeb_1(blockVec_u, j, j, j + N_0)
            alpha = j + N_0 - powerUsedSoFar
        elif j <= N - N_0:
            J_j_N = self.compute_J_j_N(j, j + N_0)
            J_j_N_param_j, J_j_N_param_N = j, j + N_0
            b = self.compute_tildeb_1(blockVec_u, N_0, j, j + N_0)
            alpha = j + N_0 - powerUsedSoFar
        else:
            J_j_N = self.compute_J_j_N(j, N)
            J_j_N_param_j, J_j_N_param_N = j, N
            b = self.compute_tildeb_1(blockVec_u, N_0, j, N)
            alpha = N - powerUsedSoFar
        return (J_j_N, J_j_N_param_j, J_j_N_param_N), b, alpha

    def compute_J_j_N_eigenvalues(self, j, N):
        if not (j == self.compute_J_j_N_eig_previous_j and N == self.compute_J_j_N_eig_previous_N):
            self.compute_J_j_N_eig_previous_j, self.compute_J_j_N_eig_previous_N = j, N
            if self.use_cuda:
                self.J_j_N_eig = torch.symeig(self.compute_J_j_N(j, N), eigenvectors=True).cuda()
            else:
                self.J_j_N_eig = torch.symeig(self.compute_J_j_N(j, N), eigenvectors=True)
        return self.J_j_N_eig


    def corollary_4_opt(self, J_j_N, tilde_b, alpha):
        batchSize, N = alpha.shape[0], J_j_N[0].shape[0]
        b = -tilde_b
        J_j_N_eig = self.compute_J_j_N_eigenvalues(J_j_N[1], J_j_N[2])

        if self.use_cuda:
            x = torch.zeros(N, 1, dtype=torch.float).cuda()
            u = torch.zeros(batchSize, N, 1, dtype=torch.float).cuda()
            lambda_star = torch.zeros(batchSize).cuda()
        else:
            x = torch.zeros(N, 1, dtype=torch.float)
            u = torch.zeros(batchSize, N, 1, dtype=torch.float)
            lambda_star = torch.zeros(batchSize)

        for batchIndex in range(batchSize):
            eigenvalues_A, eigenvectors_A = -torch.multiply(J_j_N_eig[0], torch.sqrt(alpha[batchIndex])), J_j_N_eig[1]
            # columns of eigenvectors_A are the eigenvectors
            lambda_min_A = eigenvalues_A[-1]
            if torch.max(torch.abs(b)) == 0: # pure qudratic
                lambda_star[batchIndex] = lambda_min_A
                if lambda_star[batchIndex] >= 0:
                    x.fill_(0)
                else:
                    x = eigenvectors_A[:, -1][:, None]
            else:
                eigenvectors_A_dot_b = torch.pow(torch.matmul(torch.transpose(eigenvectors_A, 1, 0), b[batchIndex]), 2)

            u[batchIndex] = torch.multiply(x, torch.sqrt(alpha[batchIndex]))

        return u, lambda_star

    def compute_caligraphE(self, u_N, tilde_e_N):
        N = int(u_N.shape[1]/self.dim_x)
        Xi_N, bar_Xi_N = self.compute_Xi_l_N(0, N), self.compute_bar_Xi_N(N)
        quadraticPart = torch.matmul(torch.transpose(u_N, 1, 2), torch.matmul(Xi_N, u_N))
        linearPart = 2*torch.matmul(torch.transpose(tilde_e_N, 1, 2), torch.matmul(bar_Xi_N, u_N))
        noPlayerPart = torch.matmul(torch.transpose(tilde_e_N, 1, 2), tilde_e_N)
        return torch.div(noPlayerPart + quadraticPart + linearPart, N)