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
from analyticResults_func import watt2db, volt2dbm, watt2dbm


def GenSysModel(dim_x, dim_z):
    if dim_x == 1:
        F = -1 + 2 * np.random.rand(dim_x, dim_x)
    else:
        F = np.random.randn(dim_x, dim_x)
        eigAbsMax = np.abs(np.linalg.eigvals(F)).max()
        F = F/((1.1+0.1*np.random.rand(1))*eigAbsMax)

    H = np.random.randn(dim_x, dim_z)
    H = H/np.linalg.norm(H)

    processNoiseVar, measurementNoiseVar = 1, 1
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

    for i in range(1, N):
        x[i] = np.matmul(F, x[i - 1]) + processNoises[i - 1]

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
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous().cuda()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.tildeF = torch.tensor(tildeF, dtype=torch.float, requires_grad=False).contiguous()
            self.tildeF_transpose = torch.tensor(tildeF.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.K = torch.tensor(K, dtype=torch.float, requires_grad=False).contiguous()
            self.H = torch.tensor(H, dtype=torch.float, requires_grad=False).contiguous()
            self.Sint = torch.tensor(Sint, dtype=torch.float, requires_grad=False).contiguous()
            self.H_transpose = torch.tensor(H.transpose(), dtype=torch.float, requires_grad=False).contiguous()
            self.thr = torch.tensor(thr, dtype=torch.float, requires_grad=False).contiguous()
            self.theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherRecursiveGain = torch.tensor(smootherRecursiveGain, dtype=torch.float, requires_grad=False).contiguous()
            self.smootherGain = torch.tensor(smootherGain, dtype=torch.float, requires_grad=False).contiguous()
            self.Ka_0 = torch.tensor(Ka_0, dtype=torch.float, requires_grad=False).contiguous()

    def forward(self, z, filterStateInit):
        # z, filterStateInit are cuda

        # filtering
        N, batchSize = z.shape[0], z.shape[1]

        if self.useCuda:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False).cuda()  #  hat_x_k_plus_1_given_k is in index [k+1]            
        else:
            hat_x_k_plus_1_given_k = torch.zeros(N, batchSize, self.dim_x, 1, dtype=torch.float, requires_grad=False)  #  hat_x_k_plus_1_given_k is in index [k+1]            

        hat_x_k_plus_1_given_k[0] = filterStateInit[0]

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

def calcTimeSeriesMeanRootEnergy(x):
    return torch.mean(torch.norm(x, dim=2), dim=0)

def calcTimeSeriesMeanEnergy(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=2), dim=0)