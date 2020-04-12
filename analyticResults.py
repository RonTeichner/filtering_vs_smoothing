import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from analyticResults_func import *

####################################################################################################################
enableFig_fig_1d_filt_f09 = False  # also fig_1d_smoothing_f09
enableFig_fig_1d_filt_f01 = False # also fig_1d_smoothing_f01
enableFig_fig_1d_filt_const_err = False
enableFig_sm_vs_fl_different_f = True

if enableFig_fig_1d_filt_f09:
    beta = 0.9  # acceleration memory
    alpha = 0.9  # velocity decay
    dt = 0.5  # sec

    F = np.array([[beta, 0], [dt, alpha]])
    H = np.array([[1], [0]])
    #G = np.eye(2)

    std_process_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
    std_meas_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)


    F = F[0:1, 0:1]
    H = H[0:1]
    deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all = calc_analytic_values(F, H, std_process_noises, std_meas_noises, firstDimOnly=True)
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True)

if enableFig_fig_1d_filt_f01:
    beta = 0.1  # acceleration memory
    alpha = 0.9  # velocity decay
    dt = 0.5  # sec

    F = np.array([[beta, 0], [dt, alpha]])
    H = np.array([[1], [0]])
    # G = np.eye(2)

    std_process_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
    std_meas_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)

    F = F[0:1, 0:1]
    H = H[0:1]
    deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all = calc_analytic_values(F, H, std_process_noises, std_meas_noises, firstDimOnly=True)
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True)

if enableFig_fig_1d_filt_const_err:
    beta = np.array([0.2, 0.9999999999])  # np.arange(0.1, 0.3, 0.1)  # acceleration memory
    alpha = 0.9  # velocity decay
    dt = 0.5  # sec

    d = 1  # single-dimension example

    nNoiseVals = 100
    std_process_noises = np.logspace(np.log10(1e-3), np.log10(1e-2), nNoiseVals, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
    std_meas_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), nNoiseVals, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)

    deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all = np.zeros((beta.size, nNoiseVals, nNoiseVals)), np.zeros((beta.size, nNoiseVals, nNoiseVals)), np.zeros((beta.size, nNoiseVals, nNoiseVals)), np.zeros((beta.size, nNoiseVals, nNoiseVals, d, d)), np.zeros((beta.size, nNoiseVals, nNoiseVals, d, d))
    for betaIdx, betaVal in enumerate(beta):
        F = np.array([[betaVal, 0], [dt, alpha]])
        H = np.array([[1], [0]])
        # G = np.eye(2)

        F = F[0:1, 0:1]
        H = H[0:1]
        deltaFS[betaIdx], E_filtering[betaIdx], E_smoothing[betaIdx], sigma_bar_all[betaIdx], sigma_j_k_all[betaIdx] = calc_analytic_values(F, H, std_process_noises, std_meas_noises, firstDimOnly=True)
    midVal = -60+20  # db
    res = 0.15  # db
    E_filtering_db = 10*np.log10(E_filtering)
    E_filtering_db[np.where(E_filtering_db > midVal + res)] = np.nan
    E_filtering_db[np.where(E_filtering_db < midVal - res)] = np.nan
    E_filtering_db[np.where(np.logical_not(np.isnan(E_filtering_db)))] = 0
    # E_filtering_db now has values around midVal only
    enable_db_Axis = True
    if enable_db_Axis:
        std_process_noises_db = 20 * np.log10(std_process_noises / std_process_noises[0])
        std_meas_noises_db = 20 * np.log10(std_meas_noises / std_meas_noises[0])
        X, Y = np.meshgrid(std_process_noises_db, std_meas_noises_db)
    else:
        X, Y = np.meshgrid(np.power(std_process_noises, 2), np.power(std_meas_noises, 2))

    fig = plt.figure()
    #ax = fig.add_subplot(111)#, projection='3d')
    Z = E_filtering_db[0].copy()
    Z[:, :] = np.nan
    for betaIdx, betaVal in enumerate(beta):
        ax = fig.add_subplot(1, 2, betaIdx+1)  # , projection='3d')
        Zbeta = E_filtering_db[betaIdx].copy()
        Z[np.where(Zbeta == 0)] = betaVal
        ax.pcolormesh(X, Y, Z, cmap=cm.gray)
        ax.grid(True)
    plt.show()
        #ax.plot_surface(X, Y, Z)
    '''
    #ax.clabel(CS, inline=1, fontsize=10)
    if enable_db_Axis:
        ax.set_ylabel('meas noise [db]')
        ax.set_xlabel('process noise [db]')
    else:
        ax.set_ylabel(r'$\sigma_v^2$')
        ax.set_xlabel(r'$\sigma_\omega^2$')
    ax.set_title(r'constant $tr(\Sigma^F)$; different $\lambda(F)$ values')
    '''
    plt.show()

if enableFig_sm_vs_fl_different_f:
    f = np.arange(0.01, 0.99, 0.01)
    fVec = np.arange(0.1, 1, 0.1)
    processNoiseVar = 1
    etaList = [0.1, 1, 10]
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    for eta in etaList:
        filteringErrorVariance_db = watt2db(steady_state_1d_filtering_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        #smoothingErrorVariance_db = watt2db(steady_state_1d_smoothing_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        plt.plot(f, filteringErrorVariance_db, label=r'$\sigma_e^2$; $\eta=%0.2f$' % eta)
        #plt.plot(f, smoothingErrorVariance_db, label=r'$\sigma_{e,s}^2$; $\eta=%0.2f$' % eta)

        varEstErr = np.zeros_like(fVec)
        for fIdx in range(fVec.size):
            fsim = fVec[fIdx]
            print(f'starting f={fsim}')
            varEstErr[fIdx], _ = simVarEst(fsim, processNoiseVar, eta)
        plt.plot(fVec, watt2db(varEstErr), linestyle='None', marker="+", color='k')
    plt.plot(fVec, watt2db(varEstErr), linestyle='None', marker="+", color='k', label='simulation')

    #plt.text(0.6, 0.75, r'$\eta=1$')
    plt.xlabel('f')
    plt.ylabel('dbW')
    plt.grid()
    plt.legend()
    plt.title(r'Filtering estimation error variances; $\sigma_\omega^2 = %d$ [dbW]' % watt2db(processNoiseVar))

    plt.subplot(1, 2, 2)
    for eta in etaList:
        #filteringErrorVariance_db = watt2db(steady_state_1d_filtering_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        smoothingErrorVariance_db = watt2db(steady_state_1d_smoothing_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        #plt.plot(f, filteringErrorVariance_db, label=r'$\sigma_e^2$; $\eta=%0.2f$' % eta)
        plt.plot(f, smoothingErrorVariance_db, label=r'$\sigma_{e,s}^2$; $\eta=%0.2f$' % eta)

        varEstErr_s = np.zeros_like(fVec)
        for fIdx in range(fVec.size):
            fsim = fVec[fIdx]
            print(f'starting f={fsim}')
            _, varEstErr_s[fIdx] = simVarEst(fsim, processNoiseVar, eta)
        plt.plot(fVec, watt2db(varEstErr_s), linestyle='None', marker="+", color='k')
    plt.plot(fVec, watt2db(varEstErr_s), linestyle='None', marker="+", color='k', label='simulation')

    # plt.text(0.6, 0.75, r'$\eta=1$')
    plt.xlabel('f')
    plt.ylabel('dbW')
    plt.grid()
    plt.legend(loc='center', bbox_to_anchor=(0.8, 0.3))
    plt.title(r'Smoothing estimation error variances; $\sigma_\omega^2 = %d$ [dbW]' % watt2db(processNoiseVar))
    plt.show()
    '''
    eta = 1
    fVec = np.arange(0.1, 1, 0.1)
    varEstErr = np.zeros_like(fVec)
    for fIdx in range(fVec.size):
        f = fVec[fIdx]
        print(f'starting f={f}')
        varEstErr[fIdx] = simVarEst(f, processNoiseVar, eta)

    plt.figure()
    plt.plot(fVec, watt2db(varEstErr), linestyle='None', marker="+")
    plt.xlabel(f)
    plt.ylabel('dbW')
    plt.grid()
    plt.show()
    '''
