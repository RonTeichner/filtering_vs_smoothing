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
        thr_db = 0.01 # db

        DeltaFirstSample = np.dot(Ka_0, np.dot(np.transpose(H), theoreticalBarSigma))
        theoreticalSmoothingFilteringDiff = solve_discrete_lyapunov(a=np.dot(theoreticalBarSigma, np.dot(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma))) , q=DeltaFirstSample)
        theoreticalSmoothingSigma = theoreticalBarSigma - theoreticalSmoothingFilteringDiff

        A_N = solve_discrete_lyapunov(a=tildeF, q=np.eye(self.dim_x))
        normalizedNoKnowledgePlayerContribution = np.trace(np.matmul(np.dot(H, np.transpose(K)), np.matmul(np.transpose(A_N), np.dot(K, np.transpose(H)))))

        smootherRecursiveGain = np.matmul(theoreticalBarSigma, np.matmul(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma)))
        smootherGain = np.linalg.inv(F) - smootherRecursiveGain

        # block matrices:
        lambda_Xi_max = list()
        N=0
        while True:
            N += 1
            lambda_Xi_max.append(np.real(np.linalg.eigvals(compute_Xi_l_N(tildeF, K, H, 0, N=N, dim_x=self.dim_x))).max())
            if N > 1000 or (N > 5 and np.abs(watt2dbm(lambda_Xi_max[-1]) - watt2dbm(lambda_Xi_max[-2])) < thr_db):
                break
        plt.plot(np.arange(1, N+1), watt2dbm(lambda_Xi_max))
        plt.ylabel('dbm')
        plt.xlabel('N')
        plt.title(r'$\lambda^{\Xi_N}_{max}$')
        plt.grid()
        plt.show()

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
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous().cuda()
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
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous()

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
    norm_x = torch.sum(torch.pow(x, 2), dim=2)
    cumsum_norm_x = torch.cumsum(norm_x, dim=0)
    return torch.div(cumsum_norm_x, torch.cumsum(torch.ones_like(cumsum_norm_x), dim=0))

def noKnowledgePlayer(N, batchSize, dim_x, sigma_u_square):
    return torch.mul(torch.sqrt(sigma_u_square), torch.randn(N, batchSize, dim_x, 1))

def noAccessPlayer(N, batchSize, sysModel, meanEnergy):
    dim_x = sysModel["F"].shape[0]
    return torch.mul(torch.sqrt(meanEnergy), torch.randn(N, batchSize, dim_x, 1))

def compute_Xi_l_N(tildeF, K, H, l, N, dim_x):
    Xi_N_l = np.zeros((N*dim_x, N*dim_x))
    K_HT = np.matmul(K, np.transpose(H))
    thr = 1e-20 * np.abs(tildeF).max()
    for r in range(N):
        for c in range(N):
            k = np.max((r, c, l - 1))
            summed = np.zeros((dim_x, dim_x))
            while True:
                k += 1
                summedItter = np.matmul(np.transpose(np.linalg.matrix_power(tildeF, k-1-r)), np.linalg.matrix_power(tildeF, k-1-c))
                summed = summed + summedItter
                if np.abs(summedItter).max() < thr:
                    break
            Xi_N_l[dim_x*r:dim_x*(r+1), dim_x*c:dim_x*(c+1)] = np.matmul(np.transpose(K_HT), np.matmul(summed, K_HT))

    return Xi_N_l
