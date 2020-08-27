import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from analyticResults_func import *

####################################################################################################################
enableFig_fig_1d_filt_f09 = False  # also fig_1d_smoothing_f09
enableFig_fig_1d_filt_f01 = False  # also fig_1d_smoothing_f01
enableFig_fig_1d_filt_const_err = False
enableFig_sm_vs_fl_different_f = False  # also \Delta_{FS}
enableFig_conclusions = False
enableUnmodeledBehaviourSim = False
enableUnmodeledBehaviourHighDimSim = True

if enableFig_conclusions:
    std_process_noises = 1
    processNoiseVar = np.power(std_process_noises, 2)
    std_meas_noises = np.logspace(np.log10(1e-1), np.log10(1e1), 1000, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
    etaVals = np.power(std_meas_noises, 2) / np.power(std_process_noises, 2)
    fVals = np.linspace(0.01, 0.99, 100)

    Delta_FS, E_filtering, E_smoothing = np.zeros((etaVals.size, fVals.size)), np.zeros((etaVals.size, fVals.size)), np.zeros((etaVals.size, fVals.size))
    for etaIdx in range(etaVals.size):
        eta = etaVals[etaIdx]
        for fIdx in range(fVals.size):
            f = fVals[fIdx]
            E_filtering[etaIdx, fIdx] = steady_state_1d_filtering_err(processNoiseVar, eta, f)
            E_smoothing[etaIdx, fIdx] = steady_state_1d_smoothing_err(processNoiseVar, eta, f)
            Delta_FS[etaIdx, fIdx] = steady_state_1d_Delta_FS(processNoiseVar, eta, f)

    etaVals_db = watt2db(etaVals)
    X, Y = np.meshgrid(fVals, etaVals_db)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
    Z = watt2dbm(E_filtering)
    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, inline=1, fontsize=8)
    ax1.set_ylabel(r'$\eta$ [db]')
    ax1.set_xlabel('f')
    ax1.set_title(r'$tr(\Sigma^F)$ [dbm]; $\sigma_\omega^2=0$ [dbW]')
    ax1.grid(True)

    Z = watt2dbm(E_smoothing)
    CS = ax2.contour(X, Y, Z)
    ax2.clabel(CS, inline=1, fontsize=8)
    #ax2.set_ylabel(r'$\eta$ [db]')
    ax2.set_xlabel('f')
    ax2.set_title(r'$tr(\Sigma^S)$ [dbm]; $\sigma_\omega^2=0$ [dbW]')
    ax2.grid(True)

    Z = Delta_FS
    CS = ax3.contour(X, Y, Z)
    ax3.clabel(CS, inline=1, fontsize=8)
    #ax3.set_ylabel(r'$\eta$ [db]')
    ax3.set_xlabel('f')
    ax3.set_title(r'$\Delta_{FS}$')
    ax3.grid(True)
    plt.show()


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
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True, with_respect_to_processNoise=False)
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True, with_respect_to_processNoise=True)

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
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True, with_respect_to_processNoise=False)
    plot_analytic_figures(std_process_noises, std_meas_noises, deltaFS, E_filtering, E_smoothing, sigma_bar_all, sigma_j_k_all, enable_db_Axis=True, with_respect_to_processNoise=True)

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
    enableSim = True
    f = np.arange(0.01, 0.99, 0.01)
    fVec = np.arange(0.1, 1, 0.1)
    processNoiseVar = 1
    etaList = [0.1, 1, 10]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for eta in etaList:
        filteringErrorVariance_db = watt2db(steady_state_1d_filtering_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        #smoothingErrorVariance_db = watt2db(steady_state_1d_smoothing_err(processNoiseVar=processNoiseVar, eta=eta, f=f))
        plt.plot(f, filteringErrorVariance_db, label=r'$\sigma_e^2$; $\eta=%0.2f$' % eta)
        #plt.plot(f, smoothingErrorVariance_db, label=r'$\sigma_{e,s}^2$; $\eta=%0.2f$' % eta)
        if enableSim:
            varEstErr = np.zeros_like(fVec)
            for fIdx in range(fVec.size):
                fsim = fVec[fIdx]
                print(f'starting f={fsim}')
                varEstErr[fIdx], _, _, _, _, _ = simVarEst(fsim, processNoiseVar, eta)
            plt.plot(fVec, watt2db(varEstErr), linestyle='None', marker="+", color='k')
    if enableSim:
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
        if enableSim:
            varEstErr_s = np.zeros_like(fVec)
            for fIdx in range(fVec.size):
                fsim = fVec[fIdx]
                print(f'starting f={fsim}')
                _, varEstErr_s[fIdx], _, _, _, _ = simVarEst(fsim, processNoiseVar, eta)
            plt.plot(fVec, watt2db(varEstErr_s), linestyle='None', marker="+", color='k')
            print(f'eta={eta}; {watt2db(varEstErr_s)}')
    if enableSim:
        plt.plot(fVec, watt2db(varEstErr_s), linestyle='None', marker="+", color='k', label='simulation')

    # plt.text(0.6, 0.75, r'$\eta=1$')
    plt.xlabel('f')
    plt.ylabel('dbW')
    plt.grid()
    plt.legend(loc='center', bbox_to_anchor=(0.8, 0.3))
    plt.title(r'Smoothing estimation error variances; $\sigma_\omega^2 = %d$ [dbW]' % watt2db(processNoiseVar))
    #plt.show()
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
    plt.figure(figsize=(6, 4))
    for eta in etaList:
        Delta_FS = steady_state_1d_Delta_FS(processNoiseVar=processNoiseVar, eta=eta, f=f)
        plt.plot(f, Delta_FS, label=r'$\Delta_{FS}$; $\eta=%0.2f$' % eta)
    plt.xlabel('f')
    plt.grid()
    plt.legend(loc='center', bbox_to_anchor=(0.2, 0.7))
    plt.title(r'$\Delta_{FS}$: Smoothing vs filtering error')
    plt.show()

if enableUnmodeledBehaviourSim:
    fVec = np.arange(0.1, 1, 0.4)
    processNoiseVar = 1
    etaList = [0.1]#, 1, 10]

    unmodeledParamsDict = {}
    unmodeledParamsDict['alpha'] = 0.75
    unmodeledParamsDict['fs'] = 10

    for eta in etaList:
        for fIdx in range(fVec.size):
            fsim = fVec[fIdx]
            # correct model performance:
            filteringErrorVariance = steady_state_1d_filtering_err(processNoiseVar=processNoiseVar, eta=eta, f=fsim)
            smoothingErrorVariance = steady_state_1d_smoothing_err(processNoiseVar=processNoiseVar, eta=eta, f=fsim)
            filteringSynthesisErrorsCorrectModel = np.sqrt(filteringErrorVariance) * np.random.randn(100000)
            smoothingSynthesisErrorsCorrectModel = np.sqrt(smoothingErrorVariance) * np.random.randn(100000)

            _, _, filteringErrors, smoothingErrors, meanPowerModeled, meanPowerUnmodeled = simVarEst(fsim, processNoiseVar, eta, unmodeledParamsDict=unmodeledParamsDict, enableUnmodeled=True)
            modeledToUnmodeled_db = watt2db(meanPowerModeled/meanPowerUnmodeled)
            plt.figure()
            n_bins = 1000
            #n, bins, patches = plt.hist(volt2dbm(np.abs(filteringSynthesisErrorsCorrectModel)), n_bins, density=True, histtype='step', cumulative=True, label=r'filtering $\alpha=0$')
            #n, bins, patches = plt.hist(volt2dbm(np.abs(smoothingSynthesisErrorsCorrectModel)), n_bins, density=True, histtype='step', cumulative=True, label=r'smoothing $\alpha=0$')
            n, bins, patches = plt.hist(volt2dbm(np.abs(filteringErrors)), n_bins, density=True, histtype='step', cumulative=True, label=r'filtering $\alpha=%0.2f$' % unmodeledParamsDict['alpha'])
            n, bins, patches = plt.hist(volt2dbm(np.abs(smoothingErrors)), n_bins, density=True, histtype='step', cumulative=True, label=r'smoothing $\alpha=%0.2f$' % unmodeledParamsDict['alpha'])
            plt.xlabel('dbm')
            plt.title(r'CDF of estimation errors; f=%0.1f; $\sigma_\omega^2=$%0.2f; $\sigma_v^2$=%0.2f; SIR=%0.2f [db]' % (fsim, processNoiseVar, eta*processNoiseVar, modeledToUnmodeled_db))
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.xlim(-20, 40)
    plt.show()

if enableUnmodeledBehaviourHighDimSim:
    np.random.seed(11) # gives unmodeled noise thr of 4.57
    #np.random.seed(9)  # gives unmodeled noise thr of -57.54

    xdim, zdim = 5, 3
    # draw F with max eigenvalue of 1
    F = np.random.randn(xdim, xdim)
    eigAbsMax = np.abs(np.linalg.eigvals(F)).max()
    F = F/(1.1*eigAbsMax)

    H = np.random.randn(xdim, zdim)
    H = H/np.linalg.norm(H)

    processNoiseVar, measurementNoiseVar = 1, 1

    traceCovFiltering, traceCovSmoothing, theoreticalTraceCovFiltering, theoreticalTraceCovSmoothing, theoreticalThresholdUnmodeledNoiseVar, unmodeledNoiseVarVec, firstMeasTraceImprovement, theoreticalFirstMeasImprove, firstMeasTraceImprovement_u, theoreticalFirstMeasImprove_u, totalSmoothingImprovement_u = simCovEst(F, H, processNoiseVar, measurementNoiseVar)

    print('Modeled: Theoretical, empirical MSE filtering: %.2f,%.2f' %(theoreticalTraceCovFiltering, traceCovFiltering))
    print('Modeled: Theoretical, empirical MSE smoothing: %.2f,%.2f' % (theoreticalTraceCovSmoothing, traceCovSmoothing))

    print('Modeled: Theoretical, empirical first future measurement MSE improvement: %.2f,%.2f' % (theoreticalFirstMeasImprove, firstMeasTraceImprovement))
    print('Modeled: Theoretical, empirical smoothing MSE improvement (all future measurements): %.2f,%.2f' % (theoreticalTraceCovFiltering-theoreticalTraceCovSmoothing, traceCovFiltering-traceCovSmoothing))

    print('Theoretical maximal unmodeled noise var: %.2f' % (theoreticalThresholdUnmodeledNoiseVar))
    if theoreticalThresholdUnmodeledNoiseVar > 0:
        unmodeledNoiseVarVec_db = watt2db(unmodeledNoiseVarVec/theoreticalThresholdUnmodeledNoiseVar)
    else:
        unmodeledNoiseVarVec_db = watt2db(unmodeledNoiseVarVec / measurementNoiseVar)

    plt.figure()
    plt.plot(unmodeledNoiseVarVec_db, firstMeasTraceImprovement_u, label=r'Empirical $\Delta^{R:j}_{j-1 \mid j}$')
    plt.plot(unmodeledNoiseVarVec_db, theoreticalFirstMeasImprove_u, '--', label=r'Theoretical $\Delta^{R:j}_{j-1 \mid j}$')
    plt.plot(unmodeledNoiseVarVec_db, totalSmoothingImprovement_u, label=r'Empirical $\Delta^{R:j}_{j-1 \mid \infty}$')
    if theoreticalThresholdUnmodeledNoiseVar > 0:
        plt.xlabel(r'$\sigma_u^2$ [db] w.r.t $\bar{\sigma}_u^2$')
    else:
        plt.xlabel(r'$\sigma_u^2$ [db] w.r.t $\sigma_v^2$')
    plt.ylabel('[W]')
    plt.grid(True)
    plt.legend()
    plt.title('Difference between filtering and smoothing MSE')
    plt.show()

    #print('Unmodeled: Theoretical, empirical first future measurement MSE improvement: %.2f,%.2f' % (theoreticalFirstMeasImprove_u, firstMeasTraceImprovement_u))
    #print('Unmodeled: Empirical smoothing MSE improvement (all future measurements): %.2f' % (totalSmoothingImprovement_u))

