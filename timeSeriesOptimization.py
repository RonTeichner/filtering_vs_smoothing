import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db, volt2dbm, watt2dbm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

enableOptimization = False
enableInvestigation = True
np.random.seed(13)
enableOnlyAngleOptimization = True

filePath = "./maximizingFiltering2D_perSampleConstrain.pt"
optimizationMode = 'maximizeFiltering' # {'maximizeFiltering', 'maximizeSmoothing', 'minimizingSmoothingImprovement'}

if enableOptimization:
    dim_x, dim_z = 2, 2
    N = 1000  # time steps
    batchSize = 64
    useCuda = False

    assert enableOnlyAngleOptimization and (dim_x == 2) or not(enableOnlyAngleOptimization)

    # estimator init values:
    filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False).contiguous()
    if useCuda:
        filterStateInit = filterStateInit.cuda()
    # filter_P_init: [1, batchSize, dim_x, dim_x]
    # filterStateInit: [1, batchSize, dim_x, 1]

    # create a single system model:
    sysModel = GenSysModel(dim_x, dim_z)
    print(f'F = {sysModel["F"]}; H = {sysModel["H"]}; Q = {sysModel["Q"]}; R = {sysModel["R"]}')
    H = torch.tensor(sysModel["H"], dtype=torch.float, requires_grad=False)
    H_transpose = torch.transpose(H, 1, 0)
    H_transpose = H_transpose.contiguous()
    if useCuda:
        H_transpose = H_transpose.cuda()

    # craete pytorch estimator:
    if optimizationMode == 'maximizeFiltering':
        pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=False, useCuda=useCuda)
    elif optimizationMode == 'maximizeSmoothing' or optimizationMode == 'minimizingSmoothingImprovement':
        pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)

    if useCuda:
        pytorchEstimator = pytorchEstimator.cuda()
    pytorchEstimator.eval()
    # create input time-series:
    if enableOnlyAngleOptimization:
        uAngle = 2*np.pi*torch.rand((N, batchSize, 1, 1), dtype=torch.float)
        #uAngle = np.pi/2 + -np.pi/8 + np.pi/4*torch.rand((N, batchSize, 1, 1), dtype=torch.float)
        #uAngle = -12.4164/180*np.pi + - np.pi/360 + np.pi/180 * torch.rand((N, batchSize, 1, 1), dtype=torch.float)
    else:
        u = torch.cat((torch.randn((N, int(batchSize/2), dim_x, 1), dtype=torch.float), 2*torch.rand((N, int(batchSize/2), dim_x, 1), dtype=torch.float)-1), dim=1)

    if useCuda:
        if enableOnlyAngleOptimization:
            uAngle = uAngle.cuda()
        else:
            u = u.cuda()
    #model = time_series_model(N, batchSize, dim_x)

    # perform an optimization:

    #optimizer = optim.SGD([u.requires_grad_()], lr=1, momentum=0.9)
    #optimizer = optim.Adadelta([u.requires_grad_()])
    if enableOnlyAngleOptimization:
        optimizer = optim.Adam([uAngle.requires_grad_()], lr=0.1)
    else:
        optimizer = optim.Adam([u.requires_grad_()], lr=1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=100, threshold=1e-6)
    meanRootInputEnergyThr = 1 # |volt|

    #uMeanNormList, uHighestNorm, uMeanNorm, epochIndex, zHighestNorm, zMeanNorm, filteringEnergyList, filteringEnergyEfficienceList, smoothingEnergyList, smoothingEnergyEfficienceList = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    epoch = -1
    displayEvery_n_epochs = 100
    startTime = time.time()
    maxEnergyEfficience, lastSaveEpoch, lastEnergyEfficienceMean = 0, 0, 0
    nEpochsWithNoSaveThr = 1000
    while True:
        epoch += 1
        #uMeanNormList.append(np.linalg.norm(u.detach().cpu().numpy(), axis=2).mean(axis=0))
        #print(f'starting epoch {epoch}')
        optimizer.zero_grad()
        pytorchEstimator.zero_grad()
        '''
        if enableInputNoise:
            with torch.no_grad():
                if optimizationMode == 'maximizeFiltering':
                    noiseStd = torch.sqrt((1/inputNoiseSNRLin) * torch.pow(u[:-1], 2).mean()) # volt
                elif optimizationMode == 'maximizeSmoothing':
                    noiseStd = torch.sqrt((1 / inputNoiseSNRLin) * torch.pow(u, 2).mean())  # volt
                #u = torch.add(u, noiseStd*torch.randn_like(u))
                u.add_(noiseStd*torch.randn_like(u))
        '''
        if enableOnlyAngleOptimization:
            u = torch.cat((torch.cos(uAngle), torch.sin(uAngle)), dim=2)
            z = torch.matmul(H_transpose, u)
        else:
            z = torch.matmul(H_transpose, u)
        x_est_f, x_est_s = pytorchEstimator(z, filterStateInit)

        if optimizationMode == 'maximizeFiltering':
            filteringMeanEnergy = calcTimeSeriesMeanEnergy(x_est_f[1:])  # mean energy at every batch [volt]
            smoothingMeanEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
            meanInputEnergy = calcTimeSeriesMeanEnergy(u[:-1])  # mean energy at every batch [volt]
            inputEnergy = torch.sum(torch.pow(u[:-1], 2), dim=2)
            #loss = torch.mean(F.relu(meanInputEnergy-meanRootInputEnergyThr) - filteringMeanEnergy)
            loss = torch.mean(torch.mean(F.relu(inputEnergy - meanRootInputEnergyThr)) - filteringMeanEnergy)
        elif optimizationMode == 'maximizeSmoothing':
            filteringMeanEnergy = calcTimeSeriesMeanEnergy(x_est_f[1:])  # mean energy at every batch [volt]
            smoothingMeanEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
            meanInputEnergy = calcTimeSeriesMeanEnergy(u)  # mean energy at every batch [volt]
            inputEnergy = torch.sum(torch.pow(u, 2), dim=2)
            #loss = torch.mean(F.relu(meanInputEnergy - meanRootInputEnergyThr) - smoothingMeanEnergy)
            loss = torch.mean(torch.mean(F.relu(inputEnergy - meanRootInputEnergyThr)) - smoothingMeanEnergy)
        elif optimizationMode == 'minimizingSmoothingImprovement':
            filteringMeanEnergy = calcTimeSeriesMeanEnergy(x_est_f[1:])  # mean energy at every batch [volt]
            smoothingMeanEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
            meanInputEnergy = calcTimeSeriesMeanEnergy(u)  # mean energy at every batch [volt]
            inputEnergy = torch.sum(torch.pow(u, 2), dim=2)
            #loss = torch.mean(F.relu(meanInputEnergy - meanRootInputEnergyThr) + (filteringMeanEnergy - smoothingMeanEnergy))
            loss = torch.mean(torch.mean(F.relu(inputEnergy - meanRootInputEnergyThr)) + (filteringMeanEnergy - smoothingMeanEnergy))

        # loss.is_contiguous()
        loss.backward()
        optimizer.step()  # parameter update

        if optimizationMode == 'maximizeFiltering':
            energyEfficience = np.divide(filteringMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy())
        elif optimizationMode == 'maximizeSmoothing':
            energyEfficience = np.divide(smoothingMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy())
        elif optimizationMode == 'minimizingSmoothingImprovement':
            energyEfficience = np.divide(smoothingMeanEnergy.detach().cpu().numpy() - filteringMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy())

        scheduler.step(energyEfficience.mean())

        if np.mod(epoch, displayEvery_n_epochs) == 0:
            if energyEfficience.max() > maxEnergyEfficience:
                maxEnergyEfficience = energyEfficience.max()
                print('SAVED epoch: %d - max,min,mean mean energy efficience %f, %f, %f; lr=%f' % (epoch, energyEfficience.max(), energyEfficience.min(), energyEfficience.mean(), scheduler._last_lr[-1]))
                pickle.dump({"sysModel": sysModel, "u": u.detach().cpu().numpy()}, open(filePath, "wb"))
            else:
                print('epoch: %d - max,min,mean mean energy efficience %f, %f, %f; lr=%f' % (epoch, energyEfficience.max(), energyEfficience.min(), energyEfficience.mean(), scheduler._last_lr[-1]))

            stopTime = time.time()
            print(f'last {displayEvery_n_epochs} epochs took {stopTime - startTime} sec')
            startTime = time.time()

        if energyEfficience.mean() > lastEnergyEfficienceMean:
            lastEnergyEfficienceMean = energyEfficience.mean()
            lastSaveEpoch = epoch

        if epoch - lastSaveEpoch > nEpochsWithNoSaveThr:
            print(f'Stoping optimization due to {epoch - lastSaveEpoch} epochs with no improvement')
            break


if enableInvestigation:
    model_results = pickle.load(open(filePath, "rb"))
    sysModel = model_results["sysModel"]
    u = model_results["u"]
    z = np.matmul(np.transpose(sysModel["H"]), u)

    theoreticalBarSigma = solve_discrete_are(a=np.transpose(sysModel["F"]), b=sysModel["H"], q=sysModel["Q"], r=sysModel["R"])
    Ka_0 = np.dot(theoreticalBarSigma, np.dot(sysModel["H"], np.linalg.inv(np.dot(np.transpose(sysModel["H"]), np.dot(theoreticalBarSigma, sysModel["H"])) + sysModel["R"])))  # first smoothing gain
    K = np.dot(sysModel["F"], Ka_0)  # steadyKalmanGain
    tildeF = sysModel["F"] - np.dot(K, np.transpose(sysModel["H"]))
    print(f'F = {sysModel["F"]}; H = {sysModel["H"]}; Q = {sysModel["Q"]}; R = {sysModel["R"]}')
    print((f'tildeF = {tildeF}; KH\' = {np.multiply(K, np.transpose(sysModel["H"]))}'))

    batchSize, dim_x = u.shape[1], sysModel["F"].shape[0]
    # run Anderson's filter & smoother:
    # estimator init values:
    filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
    x_est_f, x_est_s = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
    # x_est_f[k] has the estimation of x[k] given z[0:k-1]
    # x_est_s[k] has the estimation of x[k] given z[0:N-1]
    if optimizationMode == 'maximizeFiltering':
        objectiveMeanPowerPerBatch = np.power(x_est_f[1:], 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u[:-1], 2).sum(axis=2).mean(axis=0)
    elif optimizationMode == 'maximizeSmoothing':
        objectiveMeanPowerPerBatch = np.power(x_est_s, 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u, 2).sum(axis=2).mean(axis=0)
    elif optimizationMode == 'minimizingSmoothingImprovement':
        objectiveMeanPowerPerBatch = np.power(x_est_s, 2).sum(axis=2).mean(axis=0) - np.power(x_est_f, 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u, 2).sum(axis=2).mean(axis=0)

    objectivePowerEfficiencyPerBatch = np.divide(objectiveMeanPowerPerBatch, inputMeanPowerPerBatch)
    objectivePowerEfficiencyPerBatch_sortedIndexes = np.argsort(objectivePowerEfficiencyPerBatch, axis=0)

    maxObjectivePowerEfficiencyIndex = objectivePowerEfficiencyPerBatch_sortedIndexes[-1, 0]

    if dim_x == 2:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
    else:
        plt.figure()
    if optimizationMode == 'maximizeFiltering':
        plt.title(r'$\frac{\sum_{k=1}^{N-1} ||\xi_k||_2^2}{\sum_{k=0}^{N-2} ||x^u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
        plt.subplots_adjust(top=0.8)
        u_norm = np.linalg.norm(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(u_norm, label=r'${||x^u_k||}_2$')
        else:
            plt.plot(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$x^u_k$')

        x_est_norm = np.linalg.norm(x_est_f[1:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\xi_{k \mid k-1}||}_2$')
        else:
            plt.plot(x_est_f[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$\xi_{k \mid k-1}$')
    elif optimizationMode == 'maximizeSmoothing':
        plt.title(r'$\frac{\sum_{k=0}^{N-1} ||\xi^s_k||_2^2}{\sum_{k=0}^{N-1} ||x^u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
        plt.subplots_adjust(top=0.8)
        u_norm = np.linalg.norm(u[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(u_norm, label=r'${||x^u_k||}_2$')
        else:
            plt.plot(u[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$x^u_k$')

        x_est_norm = np.linalg.norm(x_est_s[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\xi^s_{k \mid N-1}||}_2$')
        else:
            plt.plot(x_est_s[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$\xi^s_{k \mid N-1}$')

    elif optimizationMode == 'minimizingSmoothingImprovement':
        plt.title(r'$\frac{\sum_{k=0}^{N-1} ||\xi^s_k||_2^2 - ||\xi_k||_2^2}{\sum_{k=0}^{N-1} ||x^u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
        plt.subplots_adjust(top=0.8)
        u_norm = np.linalg.norm(u[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(u_norm, label=r'${||x^u_k||}_2$')
        else:
            plt.plot(u[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$x^u_k$')

        x_est_norm = np.power(np.linalg.norm(x_est_s[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1), 2) - np.power(np.linalg.norm(x_est_f[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1), 2)
        plt.plot(x_est_norm, label=r'${||\xi^s_{k}||}_2^2 - ||\xi_{k \mid k-1}||}_2^2$')

    plt.xlabel('k')
    plt.grid()
    plt.legend()

    # plot the angles of u:
    if dim_x == 2:
        plt.subplot(1, 2, 2)
        uVec = u[:-1, maxObjectivePowerEfficiencyIndex, :, 0]
        uAngles = 180/np.pi * np.arctan(np.divide(uVec[:, 1], uVec[:, 0])) # deg
        plt.plot(uAngles, label=r'$\angle x^u_k$')
        plt.xlabel('k')
        plt.grid()
        plt.legend()

    for pIdx in range(4):
        maxObjectivePowerEfficiencyIndex = objectivePowerEfficiencyPerBatch_sortedIndexes[-1-pIdx, 0]

        enableSubPlots = False
        if enableSubPlots:
            plt.subplots(nrows=3, ncols=1)
        else:
            plt.figure()
            if optimizationMode == 'maximizeFiltering':
                plt.title(r'$\frac{\sum_{k=1}^{N-1} ||\xi_k||_2^2}{\sum_{k=0}^{N-2} ||x^u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
            elif optimizationMode == 'maximizeSmoothing':
                plt.title(r'$\frac{\sum_{k=0}^{N-1} ||\xi^s_k||_2^2}{\sum_{k=0}^{N-1} ||x^u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
            plt.subplots_adjust(top=0.8)

        if enableSubPlots: plt.subplot(4, 1, 1)
        u_norm = np.linalg.norm(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(u_norm, label=r'${||u||}_2$')
        else:
            plt.plot(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$u$')
        if enableSubPlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(u_norm, 2).mean())))
            plt.grid()
            plt.legend()

        if enableSubPlots: plt.subplot(4, 1, 2)
        z_norm = np.linalg.norm(z[:-1, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(z_norm, label=r'${||z||}_2$')
        else:
            plt.plot(z[:-1, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$z$')
        if enableSubPlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||H^{T} x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(z_norm, 2).mean())))
            plt.grid()
            plt.legend()

        if enableSubPlots: plt.subplot(4, 1, 3)
        x_est_norm = np.linalg.norm(x_est_f[1:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid k-1}||}_2$')
        else:
            plt.plot(x_est_f[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$\hat{x}_{k \mid k-1}$')
        if enableSubPlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||\xi_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(x_est_norm, 2).mean())))
            plt.grid()
            plt.legend()


        if enableSubPlots: plt.subplot(4, 1, 4)

        x_est_norm = np.linalg.norm(x_est_s[:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid N-1}||}_2$')
        else:
            plt.plot(x_est_s[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$\hat{x}_{k \mid N-1}$')
        if enableSubPlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||\xi^s_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(x_est_norm, 2).mean())))

        if not enableSubPlots:
            bar_z = z[:, maxObjectivePowerEfficiencyIndex] - np.matmul(np.transpose(sysModel["H"]), x_est_f[:, maxObjectivePowerEfficiencyIndex])
            norm_bar_z = np.linalg.norm(bar_z, axis=1)
            if dim_x > 1:
                plt.plot(norm_bar_z, label=r'$||\bar{z}||_2$')
            else:
                plt.plot(bar_z[:,0,0], label=r'$\bar{z}$')

        plt.grid()
        plt.legend()

        if enableSubPlots: plt.subplots_adjust(hspace=1)
    plt.show()