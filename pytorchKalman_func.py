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
import cvxpy as cp
from analyticResults_func import watt2db, volt2dbm, watt2dbm, calc_tildeB, calc_tildeC, calcDeltaR


def GenSysModel(dim_x, dim_z):
    if dim_x == 1:
        F = -1 + 2 * np.random.rand(dim_x, dim_x)
    else:
        F = np.random.randn(dim_x, dim_x)
        eigAbsMax = np.abs(np.linalg.eigvals(F)).max()
        F = F/((1.1+0.1*np.random.rand(1))*eigAbsMax)

    H = np.random.randn(dim_x, dim_z)
    H = H/np.linalg.norm(H)

    processNoiseVar, measurementNoiseVar = 1/dim_x, 1/dim_z
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

    return z, x, processNoises, measurementNoises

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
        A_N_directSum = calcDeltaR(a=tildeF, q=np.eye(self.dim_x))
        assert np.abs(A_N_directSum - A_N).max() < 1e-5
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

def causalPlayer(adversarialPlayersToolbox, u, processNoises, systemInitState):
    use_cuda = adversarialPlayersToolbox.use_cuda
    N, batchSize, dim_x = u.shape[0], u.shape[1], u.shape[2]

    print(f'adversarial causal player begins')

    if use_cuda:
        powerUsedSoFar = torch.zeros(batchSize, dtype=torch.float).cuda()
    else:
        powerUsedSoFar = torch.zeros(batchSize, dtype=torch.float)

    systemInitStateBlockVec = systemInitState.permute(1, 0, 2, 3).reshape(batchSize, 1 * dim_x, 1)

    for j in range(N):
        print(f'adversarial causal player begins {j+1} out of {N}')
        J_j_N = adversarialPlayersToolbox.compute_J_j_N(j, N)
        J_j_N_tupple = (J_j_N, j, N)

        if j-1 >= 0:
            u_past_blockVec = u[:j].permute(1, 0, 2, 3).reshape(batchSize, j * dim_x, 1)
        else:
            u_past_blockVec = torch.zeros(batchSize, 0, 1, dtype=torch.float)

        if j-2 >= 0:
            omega_past_blockVec = processNoises[:j-1].permute(1, 0, 2, 3).reshape(batchSize, (j-1) * dim_x, 1)
        else:
            omega_past_blockVec = torch.zeros(batchSize, 0, 1, dtype=torch.float)

        tildeb_j_N = - adversarialPlayersToolbox.compute_tildeb_j_N(u_past_blockVec, omega_past_blockVec, systemInitStateBlockVec, j, N)

        alpha = N - powerUsedSoFar

        iter_u_N_blockVec, _ = adversarialPlayersToolbox.corollary_4_opt(J_j_N=J_j_N_tupple, tilde_b=tildeb_j_N, alpha=alpha)
        iter_u = iter_u_N_blockVec.reshape(batchSize, N-j, dim_x, 1).permute(1, 0, 2, 3)
        u[j] = iter_u[0]

        powerUsedSoFar = powerUsedSoFar + torch.sum(torch.pow(u[j:j+1], 2), dim=2, keepdim=True)[0, :, 0, 0]

        print(f'adversarial causal player ends {j + 1} out of {N}')

    return u

def geniePlayer(adversarialPlayersToolbox, u, tilde_e_k_given_k_minus_1):
    use_cuda = adversarialPlayersToolbox.use_cuda
    N, batchSize, dim_x = u.shape[0], u.shape[1], u.shape[2]

    print(f'adversarial genie player begins')
    blockVec_tilde_e_full = tilde_e_k_given_k_minus_1.permute(1, 0, 2, 3).reshape(batchSize, N * dim_x, 1)

    Xi_N = adversarialPlayersToolbox.compute_Xi_l_N(0, N)
    bar_Xi_N = adversarialPlayersToolbox.compute_bar_Xi_N(N)
    # note that J_0_N = Xi_N
    J_0_N = (Xi_N, 0, N)
    if use_cuda:
        tilde_b = - torch.matmul(torch.transpose(bar_Xi_N, 1, 0), blockVec_tilde_e_full).coda()
        alpha = N * torch.ones(batchSize, dtype=torch.float).cuda()
    else:
        tilde_b = - torch.matmul(torch.transpose(bar_Xi_N, 1, 0), blockVec_tilde_e_full)
        alpha = N * torch.ones(batchSize, dtype=torch.float)

    u_N_blockVec, _ = adversarialPlayersToolbox.corollary_4_opt(J_j_N=J_0_N, tilde_b=tilde_b, alpha=alpha)
    u[:N] = u_N_blockVec.reshape(batchSize, N, dim_x, 1).permute(1, 0, 2, 3)

    return u

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
        caligraphE_Ni[:, Ni-1:Ni], _, _, _ = adversarialPlayersToolbox.compute_caligraphE(u_Ni_blockVec, blockVec_tilde_e_full)

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
    def __init__(self, pytorchEstimator, delta_u, delta_caligraphE, enableSmartPlayers):
        self.tildeF = pytorchEstimator.tildeF
        self.K = pytorchEstimator.K
        self.H = pytorchEstimator.H
        self.theoreticalBarSigma = pytorchEstimator.theoreticalBarSigma
        self.dim_x = self.tildeF.shape[0]
        self.delta_u, self.delta_caligraphE = delta_u, delta_caligraphE
        self.use_cuda = self.tildeF.is_cuda
        self.thr_db = 0.01  # db
        self.thr = 1e-20 * torch.max(torch.abs(self.tildeF))
        self.Ns_2_2N0_factor = 100  # not in use

        self.K_HT = torch.matmul(self.K, torch.transpose(self.H, 1, 0))

        self.summed = torch.zeros(self.dim_x, self.dim_x, dtype=torch.float)

        self.compute_Xi_l_N_previous_l, self.compute_Xi_l_N_previous_N = 0, 0
        self.compute_bar_Xi_N_previous_N = 0
        self.compute_bar_Xi_N_bar_Xi_N_transpose_previous = 0
        self.compute_J_j_N_previous_j, self.compute_J_j_N_previous_N = 0, 0
        self.compute_J_j_N_eig_previous_j, self.compute_J_j_N_eig_previous_N = 0, 0
        self.compute_tildeJ_j_N_previous_j, self.compute_tildeJ_j_N_previous_N = 0, 0
        self.compute_tildeJ_N0_j_N_previous_N_0, self.compute_tildeJ_N0_j_N_previous_j, self.compute_tildeJ_N0_j_N_previous_N = 0, 0, 0
        self.compute_L_j_N_previous_j, self.compute_L_j_N_previous_N = 0, 0
        self.compute_L_N0_j_N_previous_N_0, self.compute_L_N0_j_N_previous_j, self.compute_L_N0_j_N_previous_N = 0, 0, 0
        self.compute_tildeY_j_N_previous_j, self.compute_tildeY_j_N_previous_N = 0, 0

        if self.use_cuda:
            self.K_HT.cuda()
            self.summed.cuda()
            self.thr_db.cuda()
            self.thr.cuda()

        if enableSmartPlayers:
            self.compute_lambda_Xi_max(enableFigure=True)
            self.compute_lambda_bar_Xi_N_bar_Xi_N_transpose_max(enableFigure=True)

    def test_tilde_e_expression(self, systemInitState, filterStateInit, processNoises, measurementNoises, tilde_e_k_given_k_minus_1):
        N, batchSize = processNoises.shape[0], processNoises.shape[1]
        maxDiff = torch.zeros(1)
        for k in range(N):
            for j in range(N):
                tilde_e_k_given_k_minus_1_calced_at_time_j = self.compute_tilde_e_k_given_k_minus_1_calced_at_time_j(systemInitState, filterStateInit, processNoises, measurementNoises, j, k)

                currentMaxDiff = torch.sum(torch.pow(tilde_e_k_given_k_minus_1_calced_at_time_j - tilde_e_k_given_k_minus_1[k], 2), dim=2).max()  # watt
                if currentMaxDiff > maxDiff:
                    maxDiff = currentMaxDiff
                    kMaxDiff, jMaxDiff = k, j

                if j > 0:
                    currentMaxDiff = torch.sum(torch.pow(tilde_e_k_given_k_minus_1_calced_at_time_j - tilde_e_k_given_k_minus_1_calced_at_time_j_minus_1, 2), dim=2).max()  # watt
                    if currentMaxDiff > maxDiff:
                        maxDiff = currentMaxDiff
                        kMaxDiff, jMaxDiff = k, j

                tilde_e_k_given_k_minus_1_calced_at_time_j_minus_1 = tilde_e_k_given_k_minus_1_calced_at_time_j
        print(f'maxDiff is {watt2dbm(maxDiff) - watt2dbm(torch.trace(self.theoreticalBarSigma))} db w.r.t trace(bar(Sigma)); happened in k={kMaxDiff} and j={jMaxDiff}')

    def compute_tilde_e_k_given_k_minus_1_calced_at_time_j(self, systemInitState, filterStateInit, processNoises, measurementNoises, j, k):
        filterInitContribution = torch.matmul(torch.matrix_power(self.tildeF, k), systemInitState)
        summed = torch.zeros_like(filterInitContribution)
        for i in range(np.min((j-2,k-1)) + 1):
            summed = summed + torch.matmul(torch.matrix_power(self.tildeF, k-i-1), processNoises[i:i+1])
        partKnownToPlayer = filterInitContribution + summed.clone()

        summed.fill_(0)
        for i in range(np.max((0, j-1)), k):
            summed = summed + torch.matmul(torch.matrix_power(self.tildeF, k - i - 1), (processNoises[i:i + 1] - torch.matmul(self.K, measurementNoises[i:i + 1])))

        partUnknownToPlayer_01 = summed.clone()
        summed.fill_(0)
        for i in range(np.min((j - 2, k - 1)) + 1):
            summed = summed + torch.matmul(torch.matrix_power(self.tildeF, k - i - 1), torch.matmul(self.K, measurementNoises[i:i + 1]))
        partUnknownToPlayer_02 = summed.clone()
        partUnknownToPlayer_03 = torch.matmul(torch.matrix_power(self.tildeF, k), filterStateInit[None, :, :, :])

        partUnknownToPlayer = partUnknownToPlayer_01 - partUnknownToPlayer_02 - partUnknownToPlayer_03

        return partKnownToPlayer + partUnknownToPlayer

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

    def compute_bar_Xi_N_bar_Xi_N_transpose(self, N):
        if not(N == self.compute_bar_Xi_N_bar_Xi_N_transpose_previous):
            self.compute_bar_Xi_N_bar_Xi_N_transpose_previous = N
            self.bar_Xi_N_bar_Xi_N_transpose = torch.matmul(self.compute_bar_Xi_N(N), torch.transpose(self.compute_bar_Xi_N(N), 1, 0))
        return self.bar_Xi_N_bar_Xi_N_transpose


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

    def compute_tildeJ_j_N(self, j, N):
        if not(j == self.compute_tildeJ_j_N_previous_j and N == self.compute_tildeJ_j_N_previous_N):
            self.compute_tildeJ_j_N_previous_j, self.compute_tildeJ_j_N_previous_N = j, N
            self.tildeJ_j_N = self.compute_tildeJ_N0_j_N(j, j, N)
        return self.tildeJ_j_N

    def compute_L_j_N(self, j, N):
        if not(j == self.compute_L_j_N_previous_j and N == self.compute_L_j_N_previous_N):
            self.compute_L_j_N_previous_j, self.compute_L_j_N_previous_N = j, N
            self.L_j_N = self.compute_L_N0_j_N(j, j, N)
        return self.L_j_N

    def compute_L_N0_j_N(self, N_0, j, N):
        if not(N_0 == self.compute_L_N0_j_N_previous_N_0 and j == self.compute_L_N0_j_N_previous_j and N == self.compute_L_N0_j_N_previous_N):
            self.compute_L_N0_j_N_previous_N_0, self.compute_L_N0_j_N_previous_j, self.compute_L_N0_j_N_previous_N = N_0, j, N

            self.L_N0_j_N = torch.zeros((N_0-1)*self.dim_x, (N-j)*self.dim_x, dtype=torch.float)
            if self.use_cuda: self.L_N0_j_N.cuda()

            for r in range(N_0-1):
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

                    self.L_N0_j_N[self.dim_x*r:self.dim_x*(r+1), self.dim_x*c:self.dim_x*(c+1)] = torch.matmul(self.summed, self.K_HT)

        return self.L_N0_j_N

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

    def compute_tildeY_j_N(self, j, N):
        if not(j == self.compute_tildeY_j_N_previous_j and N == self.compute_tildeY_j_N_previous_N):
            self.compute_tildeY_j_N_previous_j, self.compute_tildeY_j_N_previous_N = j, N

            self.tildeY_j_N = torch.zeros(self.dim_x, (N-j)*self.dim_x, dtype=torch.float)
            if self.use_cuda: self.tildeY_j_N.cuda()

            for r in range(1):
                for c in range(N-j):
                    k = np.max((j, c-j))
                    self.summed.fill_(0)
                    if self.use_cuda:
                        summedItter = torch.tensor(float("inf")).cuda()
                    else:
                        summedItter = torch.tensor(float("inf"))
                    while True:
                        k += 1
                        if torch.max(torch.abs(summedItter)) < self.thr or k > N-1:
                            break
                        summedItter = torch.matmul(torch.transpose(torch.matrix_power(self.tildeF, k), 1, 0), torch.matrix_power(self.tildeF, k-1-c+j))
                        self.summed = self.summed + summedItter

                    self.tildeY_j_N[self.dim_x*r:self.dim_x*(r+1), self.dim_x*c:self.dim_x*(c+1)] = torch.matmul(self.summed, self.K_HT)

        return self.tildeY_j_N


    def compute_tildeb_1(self, blockVec_u, N_0, j, N):
        if blockVec_u.shape[1] == 0:
            tildeb_1 = torch.zeros_like(blockVec_u)
        else:
            tildeJ_N0_j_N = self.compute_tildeJ_N0_j_N(N_0, j, N)
            tildeb_1 = torch.matmul(torch.transpose(tildeJ_N0_j_N, 1, 0), blockVec_u)
        return tildeb_1

    def compute_tildeb_j_N(self, u, omega, systemInitState, j, N): # computing for the causal player
        # u is from time 0 up to j-1; omega is from time 0 up to j-2
        batchSize = u.shape[0]
        if j-2 >= 0:
            tildeY_j_N, tildeJ_j_N, L_j_N = self.compute_tildeY_j_N(j, N), self.compute_tildeJ_j_N(j, N), self.compute_L_j_N(j, N)
            initStateContribution = torch.matmul(torch.transpose(tildeY_j_N, 0, 1), systemInitState)
            pastActions = torch.matmul(torch.transpose(tildeJ_j_N, 0, 1), u)
            pastProccessNoises = torch.matmul(torch.transpose(L_j_N, 0, 1), omega)
            tildeb_j_N = initStateContribution + pastProccessNoises - pastActions
        elif j-1 >= 0:
            tildeY_j_N, tildeJ_j_N = self.compute_tildeY_j_N(j, N), self.compute_tildeJ_j_N(j, N)
            initStateContribution = torch.matmul(torch.transpose(tildeY_j_N, 0, 1), systemInitState)
            pastActions = torch.matmul(torch.transpose(tildeJ_j_N, 0, 1), u)
            tildeb_j_N = initStateContribution - pastActions
        else:
            tildeY_j_N = self.compute_tildeY_j_N(j, N)
            initStateContribution = torch.matmul(torch.transpose(tildeY_j_N, 0, 1), systemInitState)
            tildeb_j_N = initStateContribution

        return tildeb_j_N

    def compute_lambda_bar_Xi_N_bar_Xi_N_transpose_max(self, enableFigure):
        lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max, corresponding_eigenvector = list(), list()
        u_0 = np.zeros((1000, self.dim_x, 1))
        N = 0
        while True:
            N += 1
            e, v = np.linalg.eig(self.compute_bar_Xi_N_bar_Xi_N_transpose(N).cpu().numpy())
            maxEigenvalueIndex = np.argmax(np.real(e))
            lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max.append(np.real(e[maxEigenvalueIndex]))
            corresponding_eigenvector.append(np.sqrt(N) * np.real(v[:, maxEigenvalueIndex]))
            u_0[N - 1] = corresponding_eigenvector[N - 1][:self.dim_x][:, None]
            if N > 20:
                stepsDiff = 1  # int(N/2)
                print(f'{N}')
                if N >= 100 or \
                        (N > 20 and self.windowSizeTest(
                            torch.tensor(u_0[N - 1 - stepsDiff][None, None, :, :], dtype=torch.float),
                            torch.tensor(u_0[N - 1][None, None, :, :], dtype=torch.float),
                            torch.tensor(lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max[-1 - stepsDiff], dtype=torch.float),
                            torch.tensor(lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max[-1], dtype=torch.float)
                            , 1, 1)):
                    break

        if self.use_cuda:
            self.lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max = torch.tensor(lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max = torch.tensor(lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous()

        u_0 = u_0[:N]

        if enableFigure:
            self.lambda_max_figures(lambda_bar_Xi_N_bar_Xi_N_transpose_Xi_max, u_0, N, corresponding_eigenvector, [r'$\lambda^{\bar{\Xi}_{N}{\bar{\Xi}_{N}}''}_{max}$', r'$\frac{\lambda^{\bar{\Xi}_{N+1}{\bar{\Xi}_{N+1}}''}_{max}}{\lambda^{\bar{\Xi}_{N}{\bar{\Xi}_{N}}''}_{max}}$'])


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
                        (N > 20 and self.windowSizeTest(
                            torch.tensor(u_0[N - 1 - stepsDiff][None, None, :, :], dtype=torch.float),
                            torch.tensor(u_0[N - 1][None, None, :, :], dtype=torch.float),
                            torch.tensor(lambda_Xi_max[-1 - stepsDiff], dtype=torch.float),
                            torch.tensor(lambda_Xi_max[-1], dtype=torch.float)
                            , 1, 1)):
                    break
        if self.use_cuda:
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous().cuda()
        else:
            self.theoretical_lambda_Xi_N_max = torch.tensor(lambda_Xi_max[-1], dtype=torch.float, requires_grad=False).contiguous()

        u_0 = u_0[:N]

        if enableFigure:
            self.lambda_max_figures(lambda_Xi_max, u_0, N, corresponding_eigenvector, [r'$\lambda^{\Xi_{N}}_{max}$', r'$\frac{\lambda^{\Xi_{N+1}}_{max}}{\lambda^{\Xi_{N}}_{max}}$'])


    def lambda_max_figures(self, lambda_Xi_max, u_0, N, corresponding_eigenvector, strings):
        lambda_Xi_max = np.asarray(lambda_Xi_max)
        lambda_Xi_max_relation = np.divide(lambda_Xi_max[1:], lambda_Xi_max[:-1])
        # u_0 = np.zeros((lambda_Xi_max.shape[0], self.dim_x, 1))
        # for i in range(len(corresponding_eigenvector)):
        # u_0[i] = corresponding_eigenvector[i][:self.dim_x][:, None]
        # print(f'square norm of eigenvector for {i+1} time steps: {np.power(np.linalg.norm(corresponding_eigenvector[i]), 2)}')
        eigenvectors_energy = np.sum(np.power(u_0, 2), axis=1)
        if self.dim_x == 2:
            eigenvectors_angle_deg = np.arctan2(u_0[:, 1], u_0[:, 0]) / np.pi * 180
        eigenvectors_energy_relation = np.divide(eigenvectors_energy[1:], eigenvectors_energy[:-1])
        # delta_eigenvectors_energy = np.sum(np.power(eigenvectors[1:] - eigenvectors[:-1], 2), axis=1)
        # delta_eigenvector_energy_relation = np.divide(delta_eigenvectors_energy, np.sum(np.power(eigenvectors[:-1], 2), axis=1))
        # delta_eigenvalue_energy = lambda_Xi_max[1:] / lambda_Xi_max[:-1]

        plt.figure(figsize=(12, 4))
        plt.subplot(2, 2, 1)
        plt.plot(np.arange(1, N + 1), watt2dbm(lambda_Xi_max), label=strings[0])
        plt.ylabel('dbm')
        plt.xlabel('N')
        plt.grid()
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(np.arange(1, N + 1), watt2dbm(eigenvectors_energy),
                 label=r'$||u_{N}[0]||_2^2$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
        plt.ylabel('dbm')
        plt.xlabel('N')
        plt.legend()
        # plt.title(r'$\lambda^{\Xi_N}_{max}$')
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(np.arange(2, N + 1), watt2db(eigenvectors_energy_relation),
                 label=r'$\frac{||u_{N+1}[0]||_2^2}{||u_{N}[0]||_2^2}$')
        plt.ylabel('db')
        plt.xlabel('N')
        plt.legend()
        # plt.title(r'$\lambda^{\Xi_N}_{max}$')
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(np.arange(2, N + 1), watt2db(lambda_Xi_max_relation),
                 label=strings[1])
        plt.ylabel('db')
        plt.xlabel('N')
        plt.legend()
        # plt.title(r'$\lambda^{\Xi_N}_{max}$')
        plt.grid()

        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(wspace=0.4)

        # plotting the time-series control for longest N available
        blockVec_u = corresponding_eigenvector[-1]
        u = blockVec_u.reshape(int(blockVec_u.shape[0] / self.dim_x), self.dim_x, 1)
        u_energy = np.power(np.linalg.norm(u, axis=1, keepdims=True), 2)
        plt.figure()
        if self.dim_x == 2: plt.subplot(2, 1, 1)
        plt.plot(np.arange(0, N), watt2dbm(u_energy)[:, 0, 0], label=r'$||u_k||_2^2$')
        plt.title(f'Energy of control for {N} time-steps horizon')
        plt.xlabel('k')
        plt.ylabel('dbm')
        plt.legend()
        plt.grid()

        if self.dim_x == 2:
            u_angle_deg = np.arctan2(u[:, 1], u[:, 0]) / np.pi * 180
            plt.subplot(2, 1, 2)
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
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(1, N + 1), watt2dbm(eigenvectors_energy), '--*',
                     label=r'$||u_{N}[0]||_2^2$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
            plt.ylabel('dbm')
            plt.xlabel('N')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(np.arange(1, N + 1), eigenvectors_angle_deg, '--*',
                     label=r'$\angle u_{N}[0]$')  # =N||{v^{\Xi_{N}}_{max}}[0:n-1]||_2^2
            plt.ylabel('deg')
            plt.xlabel('N')
            plt.legend()
            plt.grid()

            plt.subplots_adjust(hspace=0.5)

        # plt.show()

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

        identicalAlphas, identical_bs = (alpha.var() == 0), (tilde_b.var(dim=0).max() == 0)
        identicalBatches = identicalAlphas and identical_bs
        for batchIndex in range(batchSize):
            if batchIndex == 0 or not(identicalAlphas):
                A = torch.multiply(-torch.sqrt(alpha[batchIndex]), J_j_N[0])
                eigenvalues_A, eigenvectors_A = -torch.multiply(J_j_N_eig[0], torch.sqrt(alpha[batchIndex])), J_j_N_eig[1]
                # columns of eigenvectors_A are the eigenvectors
                lambda_min_A = eigenvalues_A[-1]
            if torch.max(torch.abs(b)) == 0: # pure qudratic
                lambda_star[batchIndex] = lambda_min_A
                if lambda_star[batchIndex] >= 0:
                    x.fill_(0)
                else:
                    x = eigenvectors_A[:, -1][:, None]
                u[batchIndex] = torch.multiply(x, torch.sqrt(alpha[batchIndex]))
                if identicalBatches:
                    u[1:] = u[0]
                    break
            else:
                if identicalAlphas:
                    # assuming identicalAlphas: (otherwise must write some more code)
                    square_eigenvectors_A_dot_b = torch.pow(torch.matmul(torch.transpose(eigenvectors_A, 1, 0), b), 2)
                    lambda_star = self.calc_lambda_star(square_eigenvectors_A_dot_b[:, :, 0], eigenvalues_A) # calculating lambda_star for all batches
                    x = - torch.matmul(torch.inverse(A[None, :, :].repeat(batchSize, 1, 1) + torch.multiply(lambda_star[:, :, None], torch.eye(N)[None, :, :].repeat(batchSize, 1, 1))), b)
                    u = torch.multiply(x, torch.sqrt(alpha[batchIndex]))
                    break
                else:
                    square_eigenvectors_A_dot_b = torch.pow(torch.matmul(torch.transpose(eigenvectors_A, 1, 0), b[batchIndex]), 2)[None, :, :]
                    lambda_star[batchIndex] = self.calc_lambda_star(square_eigenvectors_A_dot_b[:, :, 0], eigenvalues_A)
                    x = - torch.matmul(torch.inverse(A + torch.multiply(lambda_star[batchIndex], torch.eye(N))), b[batchIndex])
                    u[batchIndex] = torch.multiply(x, torch.sqrt(alpha[batchIndex]))

        return u, lambda_star

    def calc_lambda_star(self, square_eigenvectors_A_dot_b, eigenvalues_A):
        l, step = 500, 1e-3
        batchSize = square_eigenvectors_A_dot_b.shape[0]

        if self.use_cuda:
            lambda_star, optimalVal = torch.zeros(batchSize).cuda(), torch.zeros(batchSize).cuda()
        else:
            lambda_star, optimalVal = torch.zeros(batchSize), torch.zeros(batchSize)


        lambda_min_A = eigenvalues_A[-1]
        enableBatchSolve = True
        if not(enableBatchSolve):
            # since CVXPY returns numpy:
            lambda_star_numpy, optimalVal_numpy = np.zeros((batchSize, 1)), np.zeros((batchSize, 1))
            lambdaVar = cp.Variable()
            constraints = [lambdaVar >= -lambda_min_A + 1e-10]
            for batchIdx in range(batchSize):
                print(f'convex optimization {batchIdx} out of {batchSize}')
                objective = cp.Minimize(cp.sum(cp.multiply(square_eigenvectors_A_dot_b[batchIdx], (lambdaVar + eigenvalues_A)**(-1))) + lambdaVar)
                prob = cp.Problem(objective, constraints)
                prob.solve()  # Returns the optimal value.
                lambda_star_numpy[batchIdx], optimalVal_numpy[batchIdx] = lambdaVar.value, prob.value
        else:
            lambdaBatchVar = cp.Variable((batchSize,1))
            constraintsBatch = [lambdaBatchVar >= -lambda_min_A + 1e-10]
            objectiveBatch = cp.Minimize(cp.sum(cp.sum(cp.multiply(square_eigenvectors_A_dot_b, (lambdaBatchVar + eigenvalues_A[None,:].repeat(batchSize,1))**(-1)), axis=1, keepdims=True) + lambdaBatchVar))
            probBatch = cp.Problem(objectiveBatch, constraintsBatch)
            probBatch.solve()
            lambda_star_numpy, optimalVal_numpy = lambdaBatchVar.value, probBatch.value

        if self.use_cuda:
            lambda_star, optimalVal = torch.tensor(lambda_star_numpy, dtype=torch.float).cuda(), torch.tensor(optimalVal_numpy, dtype=torch.float).cuda()
        else:
            lambda_star, optimalVal = torch.tensor(lambda_star_numpy, dtype=torch.float), torch.tensor(optimalVal_numpy, dtype=torch.float)

        enableFigure = False # plotting a few optimization functions from the batch
        if enableFigure:
            nExamples2Plot = 5
            lambdaVec = torch.arange(-eigenvalues_A.min() + step, -eigenvalues_A.min() + step + l, step)
            if self.use_cuda:
                optFuncVal = torch.zeros(nExamples2Plot, lambdaVec.shape[0], dtype=torch.float).cuda()
            else:
                optFuncVal = torch.zeros(nExamples2Plot, lambdaVec.shape[0], dtype=torch.float)

            for i, lambda_ in enumerate(lambdaVec):
                optFuncVal[:, i] = torch.sum(torch.div(square_eigenvectors_A_dot_b[:nExamples2Plot], eigenvalues_A + lambda_), dim=1) + lambda_

            plt.figure()
            for plotIdx in range(nExamples2Plot):
                plt.plot(lambdaVec, watt2dbm(optFuncVal[plotIdx]))
                optimalLambdaCurrentIdx = lambda_star[plotIdx]
                optimalValCurrentIdx = torch.sum(torch.div(square_eigenvectors_A_dot_b[plotIdx], eigenvalues_A + optimalLambdaCurrentIdx)) + optimalLambdaCurrentIdx
                plt.plot(optimalLambdaCurrentIdx, watt2dbm(optimalValCurrentIdx), 'ko')

            plt.xlabel(r'$\lambda$')
            plt.ylabel('db')
            plt.title('optimization objective examples; same system')
            plt.grid()
            plt.show()

        return lambda_star

    def compute_caligraphE(self, u_N, tilde_e_N):
        N = int(u_N.shape[1]/self.dim_x)
        Xi_N, bar_Xi_N = self.compute_Xi_l_N(0, N), self.compute_bar_Xi_N(N)
        quadraticPart = torch.matmul(torch.transpose(u_N, 1, 2), torch.matmul(Xi_N, u_N))
        linearPart = 2*torch.matmul(torch.transpose(tilde_e_N, 1, 2), torch.matmul(bar_Xi_N, u_N))
        noPlayerPart = torch.matmul(torch.transpose(tilde_e_N, 1, 2), tilde_e_N)
        return torch.div(noPlayerPart + quadraticPart + linearPart, N), torch.div(linearPart, N), torch.div(noPlayerPart, N), torch.div(quadraticPart, N)