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
filePath = "./maximizeFiltering1D.pt"

optimizationMode = 'maximizeFiltering' # {'maximizeFiltering', 'maximizeSmoothing'}

if enableOptimization:
    dim_x, dim_z = 1, 1
    N = 1000  # time steps
    batchSize = 64
    useCuda = False

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
    H = torch.tensor(sysModel["H"], dtype=torch.float, requires_grad=False)
    H_transpose = torch.transpose(H, 1, 0)
    H_transpose = H_transpose.contiguous()
    if useCuda:
        H_transpose = H_transpose.cuda()

    # craete pytorch estimator:
    if optimizationMode == 'maximizeFiltering':
        pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=False, useCuda=useCuda)
    elif optimizationMode == 'maximizeSmoothing':
        pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)

    if useCuda:
        pytorchEstimator = pytorchEstimator.cuda()
    pytorchEstimator.eval()
    # create input time-series:
    u = torch.cat((torch.randn((N, int(batchSize/2), dim_x, 1), dtype=torch.float), 2*torch.rand((N, int(batchSize/2), dim_x, 1), dtype=torch.float)-1), dim=1)
    if useCuda:
        u = u.cuda()
    #model = time_series_model(N, batchSize, dim_x)

    # perform an optimization:

    #optimizer = optim.SGD([u.requires_grad_()], lr=1, momentum=0.9)
    #optimizer = optim.Adadelta([u.requires_grad_()])
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
            loss = torch.mean(F.relu(meanInputEnergy - meanRootInputEnergyThr) - smoothingMeanEnergy)

        # loss.is_contiguous()
        loss.backward()
        optimizer.step()  # parameter update

        if optimizationMode == 'maximizeFiltering':
            energyEfficience = np.divide(filteringMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy())
        elif optimizationMode == 'maximizeSmoothing':
            energyEfficience = np.divide(smoothingMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy())

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
    # insert a guessed solution:
    u[:, 0, 0, 0] = np.ones_like(u[:, 0, 0, 0])
    
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

    objectivePowerEfficiencyPerBatch = np.divide(objectiveMeanPowerPerBatch, inputMeanPowerPerBatch)
    objectivePowerEfficiencyPerBatch_sortedIndexes = np.argsort(objectivePowerEfficiencyPerBatch, axis=0)

    maxObjectivePowerEfficiencyIndex = objectivePowerEfficiencyPerBatch_sortedIndexes[-1, 0]
    plt.figure()
    plt.title(r'$\frac{\sum_{k=1}^{N-1} ||\xi_k||_2^2}{\sum_{k=0}^{N-2} ||u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
    plt.subplots_adjust(top=0.8)
    u_norm = np.linalg.norm(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
    if dim_x > 1:
        plt.plot(u_norm, label=r'${||u||}_2$')
    else:
        plt.plot(u[:-1, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$u$')

    x_est_norm = np.linalg.norm(x_est_f[1:, maxObjectivePowerEfficiencyIndex, :, 0], axis=1)
    if dim_x > 1:
        plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid k-1}||}_2$')
    else:
        plt.plot(x_est_f[:, maxObjectivePowerEfficiencyIndex, :, 0], label=r'$\hat{x}_{k \mid k-1}$')

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
            plt.title(r'$\frac{\sum_{k=1}^{N-1} ||\xi_k||_2^2}{\sum_{k=0}^{N-2} ||u_k||_2^2}$ = %f db' % watt2db(objectivePowerEfficiencyPerBatch[maxObjectivePowerEfficiencyIndex]))
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
                plt.plot(norm_bar_z, label=r'$||\bar{z}||_2^2$')
            else:
                plt.plot(bar_z[:,0,0], label=r'$\bar{z}$')

        plt.grid()
        plt.legend()

        if enableSubPlots: plt.subplots_adjust(hspace=1)
    plt.show()