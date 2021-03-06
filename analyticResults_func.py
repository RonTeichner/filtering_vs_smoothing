import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, update, predict, batch_filter
from filterpy.common import Q_discrete_white_noise, kinematic_kf, Saver
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

def calc_sigma_bar(H, R, F, Q):
    F_t = F.transpose()
    H_t = H.transpose()
    d = F.shape[0]
    sigma_bar = np.eye(d)
    nIter = 0
    factor = 1
    while True:
        nIter += 1
        if nIter == 100000:
            factor = 20000
            print('calc_sigma_bar: factor changes')
        a = np.matmul(H_t, np.matmul(sigma_bar, H)) + R
        a_inv = np.linalg.inv(a)
        b = np.matmul(sigma_bar, np.matmul(H, np.matmul(a_inv, np.matmul(H_t, sigma_bar))))
        sigma_bar_new = np.matmul(F, np.matmul(sigma_bar - b, F_t)) + Q
        sigma_bar_ratio = np.sum(np.abs(sigma_bar_new / sigma_bar)) / F.size
        sigma_bar = sigma_bar_new
        if np.abs(sigma_bar_ratio - 1) <= factor * np.finfo(np.float).resolution:
            break
    return sigma_bar


def calc_sigma_smoothing(sigma_bar, H, F, R):
    H_t = H.transpose()
    inv_mat = np.linalg.inv(np.matmul(H_t, np.matmul(sigma_bar, H)) + R)
    K = np.matmul(F, np.matmul(sigma_bar, np.matmul(H, inv_mat)))
    F_tilde = F - np.matmul(K, H_t)
    F_tilde_t = F_tilde.transpose()
    a = np.matmul(H_t, np.matmul(sigma_bar, H)) + R
    a_inv = np.linalg.inv(a)
    core = np.matmul(H, np.matmul(a_inv, H_t))
    # i==j:
    summand = 0
    # start with s=k-j=(100-1)
    sInitRange = 100
    for s in range(sInitRange):
        summand += np.matmul(np.linalg.matrix_power(F_tilde_t, s), np.matmul(core, np.linalg.matrix_power(F_tilde, s)))
    sigma_j_k = sigma_bar - np.matmul(sigma_bar, np.matmul(summand, sigma_bar))
    # continue while sigma_j_k changes:
    s = sInitRange
    nIter = 0
    factor = 1
    while True:
        nIter += 1
        if nIter == 100000:
            factor = 20000
            print('calc_sigma_smoothing: factor changes')
        summand += np.matmul(np.linalg.matrix_power(F_tilde_t, s), np.matmul(core, np.linalg.matrix_power(F_tilde, s)))
        sigma_j_k_new = sigma_bar - np.matmul(sigma_bar, np.matmul(summand, sigma_bar))
        sigma_j_k_ratio = np.sum(np.abs(sigma_j_k_new / sigma_j_k)) / F.size
        sigma_j_k = sigma_j_k_new
        s += 1
        if np.abs(sigma_j_k_ratio - 1) <= factor * np.finfo(np.float).resolution:
            break
    return sigma_j_k

def calc_analytic_values(F, H, std_process_noises, std_meas_noises, firstDimOnly):
    F_t = F.transpose()
    H_t = H.transpose()
    d = F.shape[0]

    deltaFS, E_filtering, E_smoothing = np.zeros((std_meas_noises.size, std_process_noises.size)), np.zeros((std_meas_noises.size, std_process_noises.size)), np.zeros((std_meas_noises.size, std_process_noises.size))
    sigma_bar_all, sigma_j_k_all = np.zeros((std_meas_noises.size, std_process_noises.size, d, d)), np.zeros((std_meas_noises.size, std_process_noises.size, d, d))
    i = 0
    for pIdx, std_process_noise in enumerate(std_process_noises):
        for mIdx, std_meas_noise in enumerate(std_meas_noises):
            i += 1
            if firstDimOnly:
                Q = np.array([[np.power(std_process_noise, 2)]])
            else:
                Q = np.array([[np.power(std_process_noise, 2), 0], [0, 0]])

            R = np.power(std_meas_noise, 2)

            #  print(f'eigenvalues of F are: {np.linalg.eig(F)[0]}')

            sigma_bar = calc_sigma_bar(H, R, F, Q)
            sigma_j_k = calc_sigma_smoothing(sigma_bar, H, F, R)

            sigma_bar_all[mIdx, pIdx], sigma_j_k_all[mIdx, pIdx] = sigma_bar, sigma_j_k
            E_f = np.trace(sigma_bar)
            E_s = np.trace(sigma_j_k)

            deltaFS[mIdx, pIdx] = (E_f - E_s) / (0.5*(E_f + E_s))
            E_filtering[mIdx, pIdx], E_smoothing[mIdx, pIdx] = E_f, E_s
            #  print(f'deltaFS[pIdx, mIdx] = {deltaFS[pIdx, mIdx]}')
        print(f'finished: {100*i/(std_process_noises.size * std_meas_noises.size)} %')
    return deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all

def plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis, with_respect_to_processNoise=False):
    d = sigma_bar_all.shape[-1]

    if enable_db_Axis:
        std_process_noises_dbm = 20*np.log10(std_process_noises) + 30  #/std_process_noises[0])
        std_meas_noises_dbm = 20*np.log10(std_meas_noises) + 30  #/std_meas_noises[0])
        X, Y = np.meshgrid(std_process_noises_dbm, std_meas_noises_dbm)
    else:
        X, Y = np.meshgrid(np.power(std_process_noises, 2), np.power(std_meas_noises, 2))

    Z = deltaFS
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    if enable_db_Axis:
        ax.set_ylabel('meas noise [dbm]')
        ax.set_xlabel('process noise [dbm]')
    else:
        ax.set_ylabel(r'$\sigma_v^2$')
        ax.set_xlabel(r'$\sigma_\omega^2$')
    ax.set_title(r'$\frac{tr(\Sigma^F)-tr(\Sigma^S)}{0.5(tr(\Sigma^F)+tr(\Sigma^S))}$')
    #plt.show()

    if not(with_respect_to_processNoise):
        Z = 10*np.log10(E_filtering) + 30
        #Z = Z - Z.max()
    else:
        Z = 10*np.log10(E_filtering/np.power(std_process_noises, 2))
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    if enable_db_Axis:
        ax.set_ylabel('meas noise [dbm]')
        ax.set_xlabel('process noise [dbm]')
    else:
        ax.set_ylabel(r'$\sigma_v^2$')
        ax.set_xlabel(r'$\sigma_\omega^2$')
    if not (with_respect_to_processNoise):
        ax.set_title(r'$tr(\Sigma^F)$ [dbm]')
        plt.plot(std_process_noises_dbm, std_meas_noises_dbm, 'r--')
    else:
        ax.set_title(r'$tr(\Sigma^F)/\sigma_\omega^2$ [db]')
    plt.grid(True)
    #plt.show()

    if not (with_respect_to_processNoise):
        Z = 10*np.log10(E_smoothing) + 30
        #Z = Z - Z.max()
    else:
        Z = 10 * np.log10(E_smoothing / np.power(std_process_noises, 2))
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    if enable_db_Axis:
        ax.set_ylabel('meas noise [dbm]')
        ax.set_xlabel('process noise [dbm]')
    else:
        ax.set_ylabel(r'$\sigma_v^2$')
        ax.set_xlabel(r'$\sigma_\omega^2$')
    if not (with_respect_to_processNoise):
        ax.set_title(r'$tr(\Sigma^S)$ [dbm]')
        plt.plot(std_process_noises_dbm, std_meas_noises_dbm, 'r--')
    else:
        ax.set_title(r'$tr(\Sigma^S)/\sigma_\omega^2$ [db]')
    plt.grid(True)
    #plt.show()
    '''
    n_bins = 50
    n, bins, patches = plt.hist(E_filtering.flatten(), n_bins, density=True, histtype='step', cumulative=False, label='Ef')
    n, bins, patches = plt.hist(E_smoothing.flatten(), n_bins, density=True, histtype='step', cumulative=False, label='Es')
    plt.title('Filtering & Smoothing hist')
    plt.xlabel('trace values')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    for dIdx in range(d):
        componentVarFiltering = sigma_bar_all[:, :, dIdx, dIdx]
        componentVarSmoothing = sigma_j_k_all[:, :, dIdx, dIdx]
        component_deltaFS = (componentVarFiltering - componentVarSmoothing) / (0.5*(componentVarFiltering + componentVarSmoothing))

        Z = component_deltaFS
        Z = Z - Z.max()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        if enable_db_Axis:
            ax.set_ylabel('meas noise [dbm]')
            ax.set_xlabel('process noise [dbm]')
        else:
            ax.set_ylabel(r'$\sigma_v^2$')
            ax.set_xlabel(r'$\sigma_\omega^2$')
        ax.set_title(r'component %d: $\frac{\Sigma^F(d,d)-\Sigma^S(d,d)}{0.5(\Sigma^F(d,d)+\Sigma^S(d,d))}$ (scaled)' % (dIdx + 1))

        Z = 10 * np.log10(componentVarFiltering)
        Z = Z - Z.max()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        if enable_db_Axis:
            ax.set_ylabel('meas noise [dbm]')
            ax.set_xlabel('process noise [dbm]')
        else:
            ax.set_ylabel(r'$\sigma_v^2$')
            ax.set_xlabel(r'$\sigma_\omega^2$')
        ax.set_title(r'component %d: $\Sigma^F(d,d)$ [db]' % (dIdx+1))
        # plt.show()

        Z = 10 * np.log10(componentVarSmoothing)
        Z = Z - Z.max()
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        if enable_db_Axis:
            ax.set_ylabel('meas noise [dbm]')
            ax.set_xlabel('process noise [dbm]')
        else:
            ax.set_ylabel(r'$\sigma_v^2$')
            ax.set_xlabel(r'$\sigma_\omega^2$')
        ax.set_title(r'component %d: $\Sigma^S(d,d)$ [db]' % (dIdx+1))

    plt.show()

def steady_state_1d_filtering_err(processNoiseVar, eta, f):
    arg = eta*(np.power(f, 2) - 1) + 1
    errorVariance = 0.5*processNoiseVar*(arg + np.sqrt(np.power(arg, 2) + 4*eta))  # [W]
    return errorVariance

def steady_state_1d_smoothing_err(processNoiseVar, eta, f):
    gamma = steady_state_1d_filtering_err(processNoiseVar, eta, f) / (0.5*processNoiseVar)
    errorVariance = 0.5*processNoiseVar*(gamma - (0.5*(0.5*gamma+eta)*np.power(gamma, 2))/(np.power(0.5*gamma+eta, 2) - np.power(f*eta, 2)))
    return errorVariance

def steady_state_1d_Delta_FS(processNoiseVar, eta, f):
    gamma = steady_state_1d_filtering_err(processNoiseVar, eta, f) / (0.5 * processNoiseVar)
    arg = gamma * (0.5*gamma+eta) / (np.power(0.5*gamma+eta, 2) - np.power(f, 2)*np.power(eta, 2))
    Delta_FS = arg / (2-0.5*arg)
    return Delta_FS

def gen_1d_measurements(f, processNoiseVar, measurementNoiseVar, initState, N, unmodeledParamsDict = {}, enableUnmodeled = False):
    # unmodeled behaviour:
    unmodeledBehaiour = np.zeros((N, 1, 1))
    if enableUnmodeled:
        alpha, fs = unmodeledParamsDict['alpha'], unmodeledParamsDict['fs']
        phi_0 = np.random.rand()*(2*np.pi)
        unmodeledBehaiour[:, 0, 0] = np.sin(2*np.pi*f/fs*np.arange(0, N) + phi_0)
    else:
        alpha = 0

    # generate state
    x, z = np.zeros((N, 1, 1)), np.zeros((N, 1, 1))
    modeldPower, unmodeledPower = np.zeros(N), np.zeros(N)  # Watt
    x[0] = initState
    processNoises = np.sqrt(processNoiseVar) * np.random.randn(N)
    measurementNoises = np.sqrt(measurementNoiseVar) * np.random.randn(N)
    z[0] = x[0] + unmodeledBehaiour[0] + measurementNoises[0]
    modeldPower[0], unmodeledPower[0] = np.power(x[0], 2), np.power(unmodeledBehaiour[0,0,0], 2)
    for i in range(1, N):
        x[i] = f*x[i-1] + processNoises[i]
        z[i] = x[i] + alpha*unmodeledBehaiour[i] + measurementNoises[i]

        modeldPower[i], unmodeledPower[i] = np.power(x[i, 0, 0], 2), np.power(alpha*unmodeledBehaiour[i, 0, 0], 2)
    return x, z, modeldPower.mean(), unmodeledPower.mean()

def simVarEst(f, processNoiseVar, eta, unmodeledParamsDict = {}, enableUnmodeled = False):
    nIter = 10
    N = 10000
    measurementNoiseVar = eta / processNoiseVar
    x_err_array, x_err_s_array = np.array([]), np.array([])
    for i in range(nIter):
        k_filter = KalmanFilter(dim_x=1, dim_z=1)
        x, z, meanModeledPower, meanUnmodeledPower = gen_1d_measurements(f, processNoiseVar, measurementNoiseVar, np.sqrt(k_filter.P) * np.random.randn(1, 1), N, unmodeledParamsDict, enableUnmodeled)

        filterStateInit = np.sqrt(k_filter.P) * np.random.randn(1, 1)  # 1D only!
        k_filter.x = filterStateInit
        k_filter.Q = processNoiseVar * np.ones((1, 1))
        k_filter.R = measurementNoiseVar * np.ones((1, 1))
        k_filter.H = np.ones((1, 1))
        k_filter.F = f * np.ones((1, 1))

        # run filter:
        Fs = [k_filter.F for t in range(N)]
        Hs = [k_filter.H for t in range(N)]
        x_est, cov, _, _ = k_filter.batch_filter(z, update_first=False, Fs=Fs, Hs=Hs)  # , saver=s)
        x_est_s, _, _, _ = k_filter.rts_smoother(x_est, cov, Fs=Fs, Qs=None)
        # x_est[k] has the estimation of x[k] given z[k]. so for compatability with Anderson we should propagate x_est:
        x_est[1:] = k_filter.F * x_est[:-1]
        '''
        x_est, k_gain, x_err = np.zeros((N, 1, 1)), np.zeros((N, 1, 1)), np.zeros((N, 1, 1))
        x_est[0] = filterStateInit
        for k in range(1, N):
            k_filter.predict()
            k_filter.update(z[k-1])
            x_est[k], k_gain[k] = k_filter.x, k_filter.K
        '''
        x_err = x - x_est
        x_err_array = np.append(x_err_array, x_err[int(np.round(3 / 4 * N)):].squeeze())
        x_err_s = x - x_est_s
        x_err_s_array = np.append(x_err_s_array, x_err_s[int(np.round(3 / 8 * N)):int(np.round(5 / 8 * N))].squeeze())
        '''
        plt.plot(k_gain.squeeze()[1:])
        plt.title('kalman gain')
        plt.show()
        '''
    '''
    plt.figure()
    n_bins = 100
    n, bins, patches = plt.hist(volt2dbW(np.abs(x_err_array)), n_bins, density=True, histtype='step', cumulative=True, label='hist')
    plt.xlabel(r'$\sigma_e^2$ [dbW]')
    plt.title(r'CDF of $\sigma_e^2$; f=%0.1f' % f)
    plt.grid()
    plt.show()
    '''
    return np.var(x_err_array), np.var(x_err_s_array), x_err_array, x_err_s_array, meanModeledPower, meanUnmodeledPower

def calcDeltaR(a, q):
    dim_x = a.shape[0]
    tildeR = np.zeros((dim_x, dim_x))
    thr = 1e-20 * np.abs(a).max()
    maxValAboveThr = True
    k = 0
    while maxValAboveThr:
        a_k = np.linalg.matrix_power(a, k)
        summed = np.dot(a_k, np.dot(q, np.transpose(a_k)))
        tildeR = tildeR + summed
        k+=1
        if np.abs(summed).max() < thr:
            break
    return tildeR

def gen_measurements(F, H, Q, R, P, N):
    dim_x, dim_z = F.shape[0], H.shape[0]
    # generate state
    x, z = np.zeros((N, dim_x, 1)), np.zeros((N, dim_z, 1))

    x[0] = np.dot(np.linalg.cholesky(P), np.random.randn(dim_x, 1))

    processNoises = np.expand_dims(np.dot(np.linalg.cholesky(Q), np.random.randn(dim_x, N)).transpose(), -1)
    measurementNoises = np.expand_dims(np.dot(np.linalg.cholesky(R), np.random.randn(dim_z, N)).transpose(), -1)

    for i in range(1, N):
        x[i] = np.dot(F, x[i-1]) + processNoises[i-1]

    z = np.matmul(H, x) + measurementNoises

    return x, z

def unmodeledBehaviorSim(DeltaFirstSample, unmodeledNoiseVar, unmodeledNormalizedDecrasePerformanceMat, k_filter, N, tilde_z, filterStateInit, filter_P_init, tilde_x, nIter):
    dim_x = k_filter.F.shape[0]

    x_err_f_u_array, x_err_s_firstMeas_u_array, x_err_s_u_array = np.array([]), np.array([]), np.array([])
    # add unmodeled behavior:
    theoreticalFirstMeasImprove_u = np.trace(DeltaFirstSample) - unmodeledNoiseVar * np.trace(unmodeledNormalizedDecrasePerformanceMat)
    for i in range(nIter):
        s = np.matmul(k_filter.H, np.expand_dims(np.dot(np.linalg.cholesky(unmodeledNoiseVar * np.eye(dim_x)), np.random.randn(dim_x, N)).transpose(), -1))
        z = tilde_z + s

        # run filter on unmodeled measurement:
        k_filter.x = filterStateInit.copy()
        k_filter.P = filter_P_init.copy()
        x_est_u, cov_u, x_est_f_u, _ = k_filter.batch_filter(zs=z, update_first=False)
        x_est_s_u, _, _, _ = k_filter.rts_smoother(x_est_u, cov_u)

        x_err_f_u = np.power(np.linalg.norm(tilde_x - x_est_f_u, axis=1), 2)
        x_err_f_u_array = np.append(x_err_f_u_array, x_err_f_u[int(np.round(3 / 4 * N)):].squeeze())
        x_err_s_u = np.power(np.linalg.norm(tilde_x - x_est_s_u, axis=1), 2)
        x_err_s_u_array = np.append(x_err_s_u_array, x_err_s_u[int(np.round(3 / 8 * N)):int(np.round(5 / 8 * N))].squeeze())
        x_err_firstMeas_u = np.power(np.linalg.norm(tilde_x - x_est_u, axis=1), 2)
        x_err_s_firstMeas_u_array = np.append(x_err_s_firstMeas_u_array, x_err_firstMeas_u[int(np.round(3 / 4 * N)):].squeeze())

    traceCovFiltering_u, traceCovSmoothing_u = np.mean(x_err_f_u_array), np.mean(x_err_s_u_array)
    traceCovFirstMeas_u = np.mean(x_err_s_firstMeas_u_array)
    firstMeasTraceImprovement_u = traceCovFiltering_u - traceCovFirstMeas_u
    totalSmoothingImprovement_u = traceCovFiltering_u - traceCovSmoothing_u

    return traceCovFiltering_u, traceCovSmoothing_u, traceCovFirstMeas_u, firstMeasTraceImprovement_u, theoreticalFirstMeasImprove_u, totalSmoothingImprovement_u

def calc_tildeE(tildeF, D_int, k, i, n):
    tildeF_pow_n_minus_k = np.linalg.matrix_power(tildeF, n - k)
    tildeF_pow_n_minus_i_minus_1 = np.linalg.matrix_power(tildeF, n - i - 1)
    tildeE = np.matmul(tildeF_pow_n_minus_k.transpose(), np.matmul(D_int, tildeF_pow_n_minus_i_minus_1))

    return tildeE

def calc_tildeD(tildeF, D_int, k, i, m, N):
    dim_x = tildeF.shape[0]
    thr = 1e-20 * np.abs(tildeF).max()
    E_summed_m_to_inf = np.zeros((dim_x, dim_x))
    n = m-1
    while True:
        n += 1
        if n > N-1:
            break
        tmp = calc_tildeE(tildeF, D_int, k, i, n)
        E_summed_m_to_inf = E_summed_m_to_inf + tmp
        if np.abs(tmp).max() < thr:
            break

    return E_summed_m_to_inf

def calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i, N):
    tildeF_pow_k_minus_i_minus_1 = np.linalg.matrix_power(tildeF, k - i - 1)
    tildeB = tildeF_pow_k_minus_i_minus_1 - np.matmul(theoreticalBarSigma, calc_tildeD(tildeF, D_int, k, i, k, N))
    return tildeB

def calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, i, N):
    tildeF_pow_i_minus_k = np.linalg.matrix_power(tildeF, i - k)
    tildeC = np.matmul(theoreticalBarSigma, np.matmul(tildeF_pow_i_minus_k.transpose(), inv_F_Sigma) - calc_tildeD(tildeF, D_int, k, i, i+1, N))

    return tildeC

def recursive_calc_smoothing_anderson(z, K, H, tildeF, F, theoreticalBarSigma):  # Anderson's notations
    # time index k is from 0 to z.shape[0]
    N = z.shape[0]
    inv_F_Sigma = np.linalg.inv(np.matmul(F, theoreticalBarSigma))
    K_HT = np.matmul(K, H.transpose())
    D_int = np.matmul(inv_F_Sigma, K_HT)
    inv_tildeF = np.linalg.inv(tildeF)

    x_dim = tildeF.shape[0]
    z_dim = z.shape[1]

    # filtering, inovations:
    hat_x_k_plus_1_given_k = np.zeros((N, x_dim, 1))# hat_x_k_plus_1_given_k is in index [k+1]
    bar_z_k = np.zeros((N, z_dim, 1))
    hat_x_k_plus_1_given_k[0] = np.dot(K, z[0])
    bar_z_k[0] = z[0]
    for k in range(N-1):
        hat_x_k_plus_1_given_k[k+1] = np.dot(tildeF, hat_x_k_plus_1_given_k[k]) + np.dot(K, z[k])
    for k in range(N):
        bar_z_k[k] = z[k] - np.dot(H.transpose(), hat_x_k_plus_1_given_k[k])

    # smoothing:
    hat_x_k_given_N = np.zeros((N, x_dim, 1))
    Sint = np.matmul(np.linalg.inv(np.matmul(F, theoreticalBarSigma)), K)
    thr = 1e-20 * np.abs(tildeF).max()
    for k in range(N):
        for i in range(k, N):
            Ka_i_minus_k = np.matmul(theoreticalBarSigma, np.matmul(np.linalg.matrix_power(tildeF, i-k).transpose(), Sint))
            if i > k:
                hat_x_k_given_i = hat_x_k_given_i + np.dot(Ka_i_minus_k, bar_z_k[i])
            else:
                hat_x_k_given_i = hat_x_k_plus_1_given_k[k] + np.dot(Ka_i_minus_k, bar_z_k[i])

            if np.abs(Ka_i_minus_k).max() < thr:
                break
        hat_x_k_given_N[k] = hat_x_k_given_i

    return hat_x_k_plus_1_given_k, hat_x_k_given_N

def direct_calc_filtering(z, K, tildeF):  # Anderson's notations
    # time index k is from 0 to z.shape[0]
    N = z.shape[0]
    x_dim = tildeF.shape[0]
    thr = 1e-20 * np.abs(tildeF).max()
    x_est_f_direct_calc = np.zeros((N, x_dim, 1))  # x_est_f_direct_calc[k] has the estimation of x[k] given z[k-1]
    for k in range(N-1):
        for i in range(k+1):
            tildeF_pow_i = np.linalg.matrix_power(tildeF, i)
            tmp = np.matmul(tildeF_pow_i, np.matmul(K, z[k-i]))
            x_est_f_direct_calc[k+1] = x_est_f_direct_calc[k+1] + tmp
            if np.abs(tmp).max() < thr:
                break
    return x_est_f_direct_calc

def direct_calc_smoothing(z, K, H, tildeF, F, theoreticalBarSigma):  # Anderson's notations
    enable_B_C_expression_verification = True
    # time index k is from 0 to z.shape[0]
    N = z.shape[0]
    inv_F_Sigma = np.linalg.inv(np.matmul(F, theoreticalBarSigma))
    K_HT = np.matmul(K, H.transpose())
    D_int = np.matmul(inv_F_Sigma, K_HT)
    inv_tildeF = np.linalg.inv(tildeF)

    x_dim = tildeF.shape[0]

    y = np.matmul(K, z)

    x_est_s_direct_calc = np.zeros((N, x_dim, 1))
    if enable_B_C_expression_verification:
        B_C_FirstExpression_max, tildeD_expression_max, tildeD_futureExpression_max, tildeBC_recursive_max, initB_max = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
    for k in range(N):
        print(f'direct smoothing calc of time {k} out of {N}')
        # past measurements:
        past, future = np.zeros((x_dim, 1)), np.zeros((x_dim, 1))
        for i in range(k):
            #if not(np.mod(i,100)): print(f'direct smoothing calc of time {k} out of {N}: processing past measurement {i} out of {k}')
            tildeB = calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i, N)
            assert not(np.isnan(tildeB).any()), "tildeB is nan"
            past = past + np.matmul(tildeB, y[i])

            if enable_B_C_expression_verification:
                # check expression that exists in shifted time-series:
                B_C_FirstExpression_max[k, i] = np.abs(calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i, 10*N) - calc_tildeB(tildeF, theoreticalBarSigma, D_int, k+1, i+1, 10*N)).max()

                tildeD_k_i_k = calc_tildeD(tildeF, D_int, k, i, k, 10*N)
                tildeD_k_plus_1_i_plus_1_k_plus_1 = calc_tildeD(tildeF, D_int, k+1, i+1, k+1, 10*N)
                tildeD_expression = tildeD_k_i_k - tildeD_k_plus_1_i_plus_1_k_plus_1
                tildeD_expression_max[k, i] = np.abs(tildeD_expression).max()

                tildeB_recursive =  np.matmul(calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i, 10*N), tildeF) - calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i-1, 10*N)
                tildeBC_recursive_max[k,i] = np.abs(tildeB_recursive).max()

                if i == k-1:
                    initB = (np.eye(x_dim) - np.matmul(theoreticalBarSigma, calc_tildeD(tildeF, D_int, 0, -1, 0, 10*N))) - calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, i, 10*N)
                    initB_max[k, i] = np.abs(initB).max()

        for i in range(k, N):
            #if not(np.mod(i,100)): print(f'direct smoothing calc of time {k} out of {N}: processing future measurement {i} out of {N}')
            tildeC = calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, i, N)
            assert not (np.isnan(tildeC).any()), "tildeC is nan"
            future = future + np.matmul(tildeC, y[i])

            if enable_B_C_expression_verification:
                # check expression that exists in shifted time-series:
                B_C_FirstExpression_max[k, i] = np.abs(calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, i, 10*N) - calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k+1, i+1, 10*N)).max()

                tildeD_k_i_i_plus_1 = calc_tildeD(tildeF, D_int, k, i, i+1, 10 * N)
                tildeD_k_plus_1_i_plus_1_i_plus_2 = calc_tildeD(tildeF, D_int, k+1, i+1, i+2, 10*N)
                tildeD_futureExpression = tildeD_k_i_i_plus_1 - tildeD_k_plus_1_i_plus_1_i_plus_2
                tildeD_futureExpression_max[k, i] = np.abs(tildeD_futureExpression).max()

                if i == k:
                    tildeC_recursive = calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, k, 10*N) - np.matmul(theoreticalBarSigma, inv_F_Sigma) + np.matmul(np.matmul(theoreticalBarSigma, np.matmul(tildeF.transpose(), np.linalg.inv(theoreticalBarSigma))), (np.eye(x_dim) - calc_tildeB(tildeF, theoreticalBarSigma, D_int, k, k-1, 10*N)))
                else:
                    tildeC_recursive =  calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, i, 10*N) - np.matmul(theoreticalBarSigma, np.matmul(tildeF.transpose(), np.matmul(np.linalg.inv(theoreticalBarSigma), calc_tildeC(tildeF, theoreticalBarSigma, D_int, inv_F_Sigma, k, i-1, 10*N))))
                tildeBC_recursive_max[k,i] = np.abs(tildeC_recursive).max()

        x_est_s_direct_calc[k] = past + future

    if enable_B_C_expression_verification:
        plt.figure(figsize=(16,10))
        plt.subplot(3,2,1)
        plt.imshow(B_C_FirstExpression_max, cmap='viridis')
        plt.colorbar()
        plt.xlabel('i')
        plt.ylabel('k')
        plt.title(f'B C expressions for shifted time-series (1), max = {B_C_FirstExpression_max.max()}')

        plt.subplot(3, 2, 2)
        plt.imshow(tildeD_expression_max, cmap='viridis')
        plt.colorbar()
        plt.xlabel('i')
        plt.ylabel('k')
        plt.title(r'$max(|\tilde{D}_{k,i,k} - \tilde{D}_{k+1,i+1,k+1}|)\forall{k;i<k}$ maxVal=%f' % (tildeD_expression_max.max()))

        plt.subplot(3, 2, 4)
        plt.imshow(tildeD_futureExpression_max, cmap='viridis')
        plt.colorbar()
        plt.xlabel('i')
        plt.ylabel('k')
        plt.title(r'$max(|\tilde{D}_{k,i,i+1} - \tilde{D}_{k+1,i+1,i+2}|)\forall{k;i \geq k}$ maxVal=%f' % (tildeD_futureExpression_max.max()))

        plt.subplot(3, 2, 5)
        plt.imshow(tildeBC_recursive_max, cmap='viridis')
        plt.colorbar()
        plt.xlabel('i')
        plt.ylabel('k')
        plt.title(r'$max(|\tilde{B}_{k,i-1} - \tilde{B}_{k,i}\tilde{F}|)\forall{k;i \geq k}$; also for $\tilde{C}$ maxVal=%f' % (tildeBC_recursive_max.max()))

        plt.show()
        print(f'maxVal of initB: {initB_max.max()}')

    return x_est_s_direct_calc

def direct_calc_smoothing_eq_startSmoothingFromAllMeas(z, K, H, tildeF, F, theoreticalBarSigma):  # Anderson's notations
    # time index k is from 0 to z.shape[0]
    N = z.shape[0]
    inv_F_Sigma = np.linalg.inv(np.matmul(F, theoreticalBarSigma))
    inv_F_Sigma_mult_K = np.matmul(inv_F_Sigma, K)
    K_HT = np.matmul(K, H.transpose())
    D_int = np.matmul(inv_F_Sigma, K_HT)
    inv_tildeF = np.linalg.inv(tildeF)
    thr = 1e-20 * np.abs(tildeF).max()
    x_dim = tildeF.shape[0]

    x_est_s_direct_calc = np.zeros((N, x_dim, 1))

    for k in range(N):
        print(f'direct_calc_smoothing_eq_startSmoothingFromAllMeas: time {k} out of {N}')
        # term 1:
        term1 = np.zeros((x_dim, 1))
        i = k
        while True:
            i -= 1
            tildeF_pow_k_minus_i_minus_1 = np.linalg.matrix_power(tildeF, k - i - 1)
            tmp = np.matmul(tildeF_pow_k_minus_i_minus_1, np.matmul(K, z[i]))
            term1 = term1 + tmp
            if i <= 0 or np.abs(tildeF_pow_k_minus_i_minus_1).max() < thr:
                break

        # term 2:
        term2 = np.zeros((x_dim, 1))
        i = k-1
        while True:
            i += 1
            tildeF_pow_i_minus_k = np.linalg.matrix_power(tildeF, i-k)
            K_a_i_minus_k = np.matmul(theoreticalBarSigma, np.matmul(tildeF_pow_i_minus_k.transpose(), inv_F_Sigma_mult_K))
            tmp = np.matmul(K_a_i_minus_k, z[i])
            term2 = term2 + tmp
            if i == N-1 or np.abs(K_a_i_minus_k).max() < thr:
                break

        # term 3:
        term3 = np.zeros((x_dim, 1))
        n = k-1
        while True:
            n += 1
            if n > N-1:
                break
            tildeF_pow_n_minus_k = np.linalg.matrix_power(tildeF, n-k)
            K_a_n_minus_k = np.matmul(theoreticalBarSigma, np.matmul(tildeF_pow_n_minus_k.transpose(), inv_F_Sigma_mult_K))
            chi_n = np.zeros((x_dim, 1))
            #if n >= 1 and n < N:
            i=n
            while True:
                i -= 1
                if i < 0 or i > N-1:
                    break
                tildeF_pow_n_minus_i_minus_1 = np.linalg.matrix_power(tildeF, n - i - 1)
                tmp = np.matmul(tildeF_pow_n_minus_i_minus_1, np.matmul(K, z[i]))
                chi_n = chi_n + tmp
                if i <= 0 or np.abs(tildeF_pow_n_minus_i_minus_1).max() < thr:
                    break
            tmp = np.matmul(K_a_n_minus_k, np.matmul(H.transpose(), chi_n))
            term3 = term3 + tmp
            if np.abs(K_a_n_minus_k).max() < thr:
                break

        x_est_s_direct_calc[k] = term1 + term2 - term3

    return x_est_s_direct_calc

def simCovEst(F, H, processNoiseVar, measurementNoiseVar, enableTheoreticalResultsOnly, enableDirectVsRecursiveSmoothingDiffCheck):
    enableSanityCheckOnShiftedTimeSeries = False
    N = 300#10000
    nIterUnmodeled = 20
    uN = 30

    if enableTheoreticalResultsOnly:
        nIterUnmodeled = 1

    dim_x, dim_z = F.shape[0], H.shape[1]

    k_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    k_filter.Q = processNoiseVar * np.eye(dim_x)
    k_filter.R = measurementNoiseVar * np.eye(dim_z)
    k_filter.H = H.transpose()
    k_filter.F = F

    theoreticalBarSigma = solve_discrete_are(a=np.transpose(k_filter.F), b=np.transpose(k_filter.H), q=k_filter.Q, r=k_filter.R)
    Ka_0 = np.dot(theoreticalBarSigma, np.dot(np.transpose(k_filter.H), np.linalg.inv(np.dot(k_filter.H, np.dot(theoreticalBarSigma, np.transpose(k_filter.H))) + k_filter.R)))# first smoothing gain
    DeltaFirstSample = np.dot(Ka_0, np.dot(k_filter.H, theoreticalBarSigma))
    steadyKalmanGain = np.dot(k_filter.F, Ka_0)
    tildeF = k_filter.F - np.dot(steadyKalmanGain, k_filter.H)
    theoreticalSmoothingFilteringDiff = solve_discrete_lyapunov(a=np.dot(theoreticalBarSigma, np.dot(np.transpose(tildeF), np.linalg.inv(theoreticalBarSigma))) , q=DeltaFirstSample)
    theoreticalSmoothingSigma = theoreticalBarSigma - theoreticalSmoothingFilteringDiff
    theoreticalFirstMeasImprove = np.trace(DeltaFirstSample)

    KH_t = np.dot(steadyKalmanGain, k_filter.H)
    tildeR = solve_discrete_lyapunov(a=tildeF, q=np.dot(KH_t, np.transpose(KH_t)))
    tildeR_directSum = calcDeltaR(a=tildeF, q=np.dot(KH_t, np.transpose(KH_t)))
    assert np.abs(tildeR_directSum - tildeR).max() < 1e-5

    # check smoothing on a series that is shifted by a single time-instance equations:
    inv_F_Sigma = np.linalg.inv(np.matmul(k_filter.F, theoreticalBarSigma))
    K_HT = np.matmul(steadyKalmanGain, k_filter.H.transpose().transpose())
    inv_F_Sigma_mult_K_HT = np.matmul(inv_F_Sigma, K_HT)
    tildeR_directSum = calcDeltaR(a=tildeF.transpose(), q=inv_F_Sigma_mult_K_HT)
    diff = tildeR_directSum - np.matmul(np.linalg.inv(tildeF).transpose(), np.matmul(tildeR_directSum, tildeF)) - inv_F_Sigma_mult_K_HT
    #assert np.abs(diff).max() < 1e-5


    Ka_0H_t = np.dot(Ka_0, k_filter.H)
    unmodeledNormalizedDecrasePerformanceMat = np.dot(Ka_0H_t, np.dot(tildeR + np.eye(dim_x), np.transpose(Ka_0H_t))) - (np.dot(Ka_0H_t, tildeR) + np.dot(tildeR, np.transpose(Ka_0H_t)))

    theoreticalThresholdUnmodeledNoiseVar = np.trace(DeltaFirstSample) / np.trace(unmodeledNormalizedDecrasePerformanceMat)

    if theoreticalThresholdUnmodeledNoiseVar > 0:
        unmodeledNoiseVarVec = np.logspace(np.log10(1e-2 * theoreticalThresholdUnmodeledNoiseVar), np.log10(10 * theoreticalThresholdUnmodeledNoiseVar), uN, base=10)
    else:
        unmodeledNoiseVarVec = np.logspace(np.log10(1e-2 * np.abs(theoreticalThresholdUnmodeledNoiseVar)), np.log10(10 * np.abs(theoreticalThresholdUnmodeledNoiseVar)), uN, base=10)

    x_err_f_array, x_err_s_array, x_err_s_firstMeas_array = np.array([]), np.array([]), np.array([])

    filter_P_init = k_filter.P.copy()
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.random.randn(dim_x, 1))

    if enableTheoreticalResultsOnly:
        enableDirectFormInvestigation = False

        if enableDirectFormInvestigation:
            # investigate the direct form:
            thr = 1e-10 * np.abs(tildeF).max()

            inv_F_Sigma = np.linalg.inv(np.matmul(k_filter.F, theoreticalBarSigma))
            K_HT = np.matmul(steadyKalmanGain, k_filter.H.transpose().transpose())
            D_int = np.matmul(inv_F_Sigma, K_HT)
            tildeB_k_k_minus_1 = np.eye(dim_x) - np.matmul(theoreticalBarSigma, calc_tildeD(tildeF, D_int, 0, -1, 0, 100000))
            eigenValues, eigenVectors = np.linalg.eig(tildeB_k_k_minus_1)
            idx = eigenValues.argsort()[::-1]
            Bw = eigenValues[idx]
            Bv = eigenVectors[:, idx]

            tildeB_k_k_minus_2 = np.matmul(tildeB_k_k_minus_1, tildeF)
            eigenValues, eigenVectors = np.linalg.eig(tildeB_k_k_minus_2)
            idx = eigenValues.argsort()[::-1]
            Bw_2 = eigenValues[idx]
            Bv_2 = eigenVectors[:, idx]

            tildeB_k_k_minus_3 = np.matmul(tildeB_k_k_minus_2, tildeF)
            eigenValues, eigenVectors = np.linalg.eig(tildeB_k_k_minus_3)
            idx = eigenValues.argsort()[::-1]
            Bw_3 = eigenValues[idx]
            Bv_3 = eigenVectors[:, idx]

            tildeB_k_k_minus_4 = np.matmul(tildeB_k_k_minus_3, tildeF)
            eigenValues, eigenVectors = np.linalg.eig(tildeB_k_k_minus_4)
            idx = eigenValues.argsort()[::-1]
            Bw_4 = eigenValues[idx]
            Bv_4 = eigenVectors[:, idx]

            tildeB_k_k_minus_5 = np.matmul(tildeB_k_k_minus_4, tildeF)
            eigenValues, eigenVectors = np.linalg.eig(tildeB_k_k_minus_5)
            idx = eigenValues.argsort()[::-1]
            Bw_5 = eigenValues[idx]
            Bv_5 = eigenVectors[:, idx]

            C_k_k = np.matmul(theoreticalBarSigma, inv_F_Sigma - np.matmul(tildeF.transpose(), np.matmul(np.linalg.inv(theoreticalBarSigma), np.eye(dim_x) - tildeB_k_k_minus_1)))
            C_k_k_second_for_sanity = np.matmul(theoreticalBarSigma, inv_F_Sigma - np.matmul(tildeF.transpose(), calc_tildeD(tildeF, D_int, 0, -1, 0, 100000)))
            assert np.abs(C_k_k_second_for_sanity - C_k_k).max() < thr, 'C_k_k problem'
            eigenValues, eigenVectors = np.linalg.eig(C_k_k)
            idx = eigenValues.argsort()[::-1]
            Cw = eigenValues[idx]
            Cv = eigenVectors[:, idx]


            plt.figure()
            origin = [0, 0]
            plt.grid()

            '''
            maxVal = max(np.maximum(*np.abs([Bw, Cw])))
            plt.xlim([-maxVal, maxVal])
            plt.ylim([-maxVal, maxVal])
            plt.quiver(*origin, *Bv[:, 0],  angles='xy', scale_units='xy', scale=1 / np.abs(Bw[0]), color='g', label=r'$\tildeB_{k,k-1}$')
            plt.quiver(*origin, *Bv[:, 1], angles='xy', scale_units='xy', scale=1 / np.abs(Bw[1]), color='g')
    
            plt.quiver(*origin, *Cv[:, 0],  angles='xy', scale_units='xy', scale=1 / np.abs(Cw[0]), color='b', label=r'$\tildeC_{k,k}$')
            plt.quiver(*origin, *Cv[:, 1], angles='xy', scale_units='xy', scale=1 / np.abs(Cw[1]), color='b')
    
            plt.title(r'Eigenvectors with $||v_i||_2=\lambda_i$')
            '''
            maxVal = 1
            plt.xlim([-maxVal, maxVal])
            plt.ylim([-maxVal, maxVal])

            plt.quiver(*origin, *Bv[:, 0],  angles='xy', scale_units='xy', scale=1, color='g', label=r'$\tildeB_{k,k-1}$')
            plt.quiver(*origin, *Bv[:, 1], angles='xy', scale_units='xy', scale=1, color='g')

            plt.quiver(*origin, *Cv[:, 0],  angles='xy', scale_units='xy', scale=1, color='b', label=r'$\tildeC_{k,k}$')
            plt.quiver(*origin, *Cv[:, 1], angles='xy', scale_units='xy', scale=1, color='b')

            plt.title(r'Eigenvectors')
            plt.legend()

            plt.figure()
            origin = [0, 0]
            plt.grid()
            maxVal = 1
            plt.xlim([-maxVal, maxVal])
            plt.ylim([-maxVal, maxVal])

            plt.quiver(*origin, *Bv[:, 0], angles='xy', scale_units='xy', scale=1, color='g', label=r'$\tildeB_{k,k-1}$')
            #plt.quiver(*origin, *Bv[:, 1], angles='xy', scale_units='xy', scale=1, color='g')
            plt.quiver(*origin, *Bv_2[:, 0], angles='xy', scale_units='xy', scale=1, color='b', label=r'$\tildeB_{k,k-2}$')
            #plt.quiver(*origin, *Bv_2[:, 1], angles='xy', scale_units='xy', scale=1, color='b')
            plt.quiver(*origin, *Bv_3[:, 0], angles='xy', scale_units='xy', scale=1, color='r', label=r'$\tildeB_{k,k-3}$')
            #plt.quiver(*origin, *Bv_3[:, 1], angles='xy', scale_units='xy', scale=1, color='r')

            plt.quiver(*origin, *Bv_4[:, 0], angles='xy', scale_units='xy', scale=1, color='k', label=r'$\tildeB_{k,k-4}$')
            plt.quiver(*origin, *Bv_5[:, 0], angles='xy', scale_units='xy', scale=1, color='m', label=r'$\tildeB_{k,k-5}$')

            plt.legend()

            plt.show()




    if not enableTheoreticalResultsOnly:
        enableFilterAdversarialInvestigation = True

        tilde_x, tilde_z = gen_measurements(k_filter.F, k_filter.H, k_filter.Q, k_filter.R, k_filter.P, N)

        enableFilterAdversarialInvestigation = True
        if enableFilterAdversarialInvestigation:
           # run the filter on adversarial optimal time-series
            x=3

        # run filter on modeled measurement:
        k_filter.x = filterStateInit.copy()
        k_filter.P = filter_P_init.copy()
        x_est, cov, x_est_f, _ = k_filter.batch_filter(zs=tilde_z, update_first=False)
        x_est_s, _, _, _ = k_filter.rts_smoother(x_est, cov)
        # x_est[k] has the estimation of x[k] given z[k]. so for compatability with Anderson we should propagate x_est:
        # x_est[1:] = k_filter.F * x_est[:-1]
        # x_est_f is compatible with Anderson ==> x_est_f[k] has the estimation of x[k] given z[k-1]

        if enableDirectVsRecursiveSmoothingDiffCheck:
            # compare smoothing estimation to a direct (not recursive) calculation
            x_est_f_direct_calc = direct_calc_filtering(tilde_z, steadyKalmanGain, tildeF)
            # x_est_s_direct_calc_eq_startSmoothingFromAllMeas = direct_calc_smoothing_eq_startSmoothingFromAllMeas(tilde_z, steadyKalmanGain, k_filter.H.transpose(), tildeF, k_filter.F, theoreticalBarSigma)
            x_est_s_direct_calc = direct_calc_smoothing(tilde_z, steadyKalmanGain, k_filter.H.transpose(), tildeF, k_filter.F, theoreticalBarSigma)
            x_est_f_recursive_calc, x_est_s_recursive_calc = recursive_calc_smoothing_anderson(tilde_z, steadyKalmanGain, k_filter.H.transpose(), tildeF, k_filter.F, theoreticalBarSigma)

            if enableSanityCheckOnShiftedTimeSeries:
                # sanity check: direct calc on shifted time-series:
                shifted_tilde_z = np.concatenate((np.random.rand(1, dim_z, 1), tilde_z[:-1]), axis=0) # shifted_tilde_z[k] = tilde_z[k-1]
                x_est_s_direct_calc_on_shifted = direct_calc_smoothing(shifted_tilde_z, steadyKalmanGain, k_filter.H.transpose(), tildeF, k_filter.F, theoreticalBarSigma)
                smoothing_shiftedDirect_direct_diff_energy = np.power(np.linalg.norm(x_est_s_direct_calc_on_shifted[1:] - x_est_s_direct_calc[:-1], axis=1), 2)
                plt.figure()
                plt.plot(smoothing_shiftedDirect_direct_diff_energy, label='DirectVsShiftedDirect')
                plt.title(r'Smoothing: direct vs shiftedDirect diff')
                plt.ylabel('W')
                plt.legend()
                plt.grid()
                plt.show()


            filtering_recursiveSimon_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_f_direct_calc - x_est_f, axis=1), 2), np.power(np.linalg.norm(x_est_f, axis=1), 2)))
            filtering_recursiveAnderson_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_f_direct_calc - x_est_f_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_f, axis=1), 2)))
            filtering_recursiveAnderson_recursiveSimon_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_f - x_est_f_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_f, axis=1), 2)))

            #smoothing_eq_startSmoothingFromAllMeas_recursiveSimon_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s_direct_calc_eq_startSmoothingFromAllMeas - x_est_s, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))
            #smoothing_eq_startSmoothingFromAllMeas_recursiveAnderson_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s_direct_calc_eq_startSmoothingFromAllMeas - x_est_s_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))
            smoothing_eq_startSmoothingFromAllMeas_recursiveAnderson_recursiveSimon_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s - x_est_s_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))

            smoothing_recursiveSimon_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s_direct_calc - x_est_s, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))
            smoothing_recursiveAnderson_direct_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s_direct_calc - x_est_s_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))
            smoothing_recursiveAnderson_recursiveSimon_diff_energy = watt2db(np.divide(np.power(np.linalg.norm(x_est_s - x_est_s_recursive_calc, axis=1), 2), np.power(np.linalg.norm(x_est_s, axis=1), 2)))

            plt.figure(figsize=(16, 8))
            plt.subplot(3, 3, 1)
            plt.plot(filtering_recursiveSimon_direct_diff_energy, label='DirectVsSimon')
            plt.title(r'Filtering: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 4)
            plt.plot(filtering_recursiveAnderson_direct_diff_energy, label='DirectVsAnderson')
            #plt.title(r'Filtering: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 7)
            plt.plot(filtering_recursiveAnderson_recursiveSimon_diff_energy, label='SimonVsAnderson')
            #plt.title(r'Filtering: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 2)
            #plt.plot(smoothing_eq_startSmoothingFromAllMeas_recursiveSimon_direct_diff_energy, label='DirectVsSimon')
            plt.title(r'Smoothing: eq_startSmoothingFromAllMeas vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 5)
            #plt.plot(smoothing_eq_startSmoothingFromAllMeas_recursiveAnderson_direct_diff_energy, label='DirectVsAnderson')
            plt.title(r'Smoothing: eq_startSmoothingFromAllMeas vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 8)
            plt.plot(smoothing_eq_startSmoothingFromAllMeas_recursiveAnderson_recursiveSimon_diff_energy, label='SimonVsAnderson')
            #plt.title(r'Smoothing: eq_startSmoothingFromAllMeas vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 3)
            plt.plot(smoothing_recursiveSimon_direct_diff_energy, label='DirectVsSimon')
            plt.title(r'Smoothing: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 6)
            plt.plot(smoothing_recursiveAnderson_direct_diff_energy, label='DirectVsAnderson')
            #plt.title(r'Smoothing: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()

            plt.subplot(3, 3, 9)
            plt.plot(smoothing_recursiveAnderson_recursiveSimon_diff_energy, label='SimonVsAnderson')
            #plt.title(r'Smoothing: direct vs recursive diff')
            plt.ylabel('db')
            plt.legend()
            plt.grid()
            plt.show()

        x_err_f = np.power(np.linalg.norm(tilde_x - x_est_f, axis=1), 2)
        x_err_f_array = np.append(x_err_f_array, x_err_f[int(np.round(3 / 4 * N)):].squeeze())
        x_err_s = np.power(np.linalg.norm(tilde_x - x_est_s, axis=1), 2)
        x_err_s_array = np.append(x_err_s_array, x_err_s[int(np.round(3 / 8 * N)):int(np.round(5 / 8 * N))].squeeze())
        x_err_firstMeas = np.power(np.linalg.norm(tilde_x - x_est, axis=1), 2)
        x_err_s_firstMeas_array = np.append(x_err_s_firstMeas_array, x_err_firstMeas[int(np.round(3 / 4 * N)):].squeeze())
    else:
        tilde_x, tilde_z = 0, 0

    traceCovFiltering, traceCovSmoothing = np.mean(x_err_f_array), np.mean(x_err_s_array)
    theoreticalTraceCovFiltering, theoreticalTraceCovSmoothing = np.trace(theoreticalBarSigma), np.trace(theoreticalSmoothingSigma)
    traceCovFirstMeas = np.mean(x_err_s_firstMeas_array)
    firstMeasTraceImprovement = traceCovFiltering - traceCovFirstMeas

    uN = unmodeledNoiseVarVec.shape[0]
    traceCovFiltering_u, traceCovSmoothing_u, traceCovFirstMeas_u, firstMeasTraceImprovement_u, theoreticalFirstMeasImprove_u, totalSmoothingImprovement_u = np.zeros(uN), np.zeros(uN), np.zeros(uN), np.zeros(uN), np.zeros(uN), np.zeros(uN)
    for uIdx, unmodeledNoiseVar in enumerate(unmodeledNoiseVarVec):
        traceCovFiltering_u[uIdx], traceCovSmoothing_u[uIdx], traceCovFirstMeas_u[uIdx], firstMeasTraceImprovement_u[uIdx], theoreticalFirstMeasImprove_u[uIdx], totalSmoothingImprovement_u[uIdx] = unmodeledBehaviorSim(DeltaFirstSample, unmodeledNoiseVar, unmodeledNormalizedDecrasePerformanceMat, k_filter, N, tilde_z, filterStateInit, filter_P_init, tilde_x, nIterUnmodeled)
        print(f'finished unmodeled var no. {uIdx} out of {unmodeledNoiseVarVec.shape[0]}')

    return traceCovFiltering, traceCovSmoothing, theoreticalTraceCovFiltering, theoreticalTraceCovSmoothing, theoreticalThresholdUnmodeledNoiseVar, unmodeledNoiseVarVec, firstMeasTraceImprovement, theoreticalFirstMeasImprove, firstMeasTraceImprovement_u, theoreticalFirstMeasImprove_u, totalSmoothingImprovement_u

def dbm2var(x_dbm):
    return np.power(10, np.divide(x_dbm - 30, 10))

def volt2dbm(x_volt):
    return 10*np.log10(np.power(x_volt, 2)) + 30

def volt2dbW(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def volt2db(x_volt):
    return 10*np.log10(np.power(x_volt, 2))

def watt2dbm(x_volt):
    return 10*np.log10(x_volt) + 30

def watt2db(x_volt):
    return 10*np.log10(x_volt)