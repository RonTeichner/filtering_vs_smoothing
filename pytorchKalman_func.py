import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
from filterpy.common import Q_discrete_white_noise, kinematic_kf, Saver
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
import torch

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
    dim_x, dim_z = F.shape[0], H.shape[1]
    N, batchSize = z.shape[0], z.shape[1]

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
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float).cuda()
    z = torch.tensor(z, dtype=torch.float).cuda()
    H = torch.tensor(H, dtype=torch.float).cuda()
    Sint = torch.tensor(Sint, dtype=torch.float).cuda()
    H_transpose = torch.transpose(H, 1, 0)
    thr = torch.tensor(thr, dtype=torch.float).cuda()
    theoreticalBarSigma = torch.tensor(theoreticalBarSigma, dtype=torch.float).cuda()

    # filtering, inovations:
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