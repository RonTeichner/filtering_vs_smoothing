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

enableOptimization = True
enableInvestigation = True
filePath = "./maximizeFiltering1D.pt"

optimizationMode = 'maximizeFiltering' # {'maximizeFiltering', 'maximizeSmoothing'}

if enableOptimization:
    #np.random.seed(11)
    dim_x, dim_z = 1, 1
    N = 1000  # time steps
    batchSize = 32
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
    u = torch.randn((N, batchSize, dim_x, 1), dtype=torch.float)
    if useCuda:
        u = u.cuda()
    #model = time_series_model(N, batchSize, dim_x)

    # perform an optimization:

    #optimizer = optim.SGD([u.requires_grad_()], lr=0.001)
    #optimizer = optim.Adadelta([u.requires_grad_()])
    optimizer = optim.Adam([u.requires_grad_()], lr=0.01)
    meanRootInputEnergyThr = 1 # |volt|

    uMeanNormList, uHighestNorm, uMeanNorm, epochIndex, zHighestNorm, zMeanNorm, filteringEnergyList, filteringEnergyEfficienceList, smoothingEnergyList, smoothingEnergyEfficienceList = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    epoch = -1
    displayEvery_n_epochs = 100
    startTime = time.time()
    maxFilterEnergyEfficience = 0
    while True:
        epoch += 1
        uMeanNormList.append(np.linalg.norm(u.detach().cpu().numpy(), axis=2).mean(axis=0))
        #print(f'starting epoch {epoch}')
        optimizer.zero_grad()
        pytorchEstimator.zero_grad()
        z = torch.matmul(H_transpose, u)
        x_est_f, x_est_s = pytorchEstimator(z, filterStateInit)

        if optimizationMode == 'maximizeFiltering':
            filteringMeanEnergy = calcTimeSeriesMeanEnergy(x_est_f[1:])  # mean energy at every batch [volt]
            smoothingMeanEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
            meanInputEnergy = calcTimeSeriesMeanEnergy(u[:-1])  # mean energy at every batch [volt]
            loss = torch.mean(F.relu(meanInputEnergy-meanRootInputEnergyThr) - filteringMeanEnergy)
        elif optimizationMode == 'maximizeSmoothing':
            filteringMeanEnergy = calcTimeSeriesMeanEnergy(x_est_f[1:])  # mean energy at every batch [volt]
            smoothingMeanEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
            meanInputEnergy = calcTimeSeriesMeanEnergy(u)  # mean energy at every batch [volt]
            loss = torch.mean(F.relu(meanInputEnergy - meanRootInputEnergyThr) - smoothingMeanEnergy)

        # loss.is_contiguous()
        loss.backward()
        optimizer.step()  # parameter update
        '''
        if np.mod(epoch, displayEvery_n_epochs) == 0:
            stopTime = time.time()
            print(f'epoch {epoch}; last {displayEvery_n_epochs} epochs took {stopTime - startTime} sec')
            startTime = time.time()
        '''
        filteringEnergyList.append(filteringMeanEnergy)
        filteringEnergyEfficienceList.append(np.divide(filteringMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy()))

        smoothingEnergyList.append(smoothingMeanEnergy)
        smoothingEnergyEfficienceList.append(np.divide(smoothingMeanEnergy.detach().cpu().numpy(), meanInputEnergy.detach().cpu().numpy()))


        if np.mod(epoch, displayEvery_n_epochs) == 0:
            uHighestNorm.append(np.linalg.norm(u.detach().cpu().numpy()[:-1], axis=2).max())
            uMeanNorm.append(np.linalg.norm(u.detach().cpu().numpy()[:-1], axis=2).mean())
            zHighestNorm.append(np.linalg.norm(torch.matmul(H_transpose, u).detach().cpu().numpy()[:-1], axis=2).max())
            zMeanNorm.append(np.linalg.norm(torch.matmul(H_transpose, u).detach().cpu().numpy()[:-1], axis=2).mean())
            epochIndex.append(epoch)
            if optimizationMode == 'maximizeFiltering':
                print('epoch: %d - max,min,mean mean filtering energy efficience %f, %f, %f' % (epoch, filteringEnergyEfficienceList[-1].max(), filteringEnergyEfficienceList[-1].min(), filteringEnergyEfficienceList[-1].mean()))
            elif optimizationMode == 'maximizeSmoothing':
                print('epoch: %d - max,min,mean mean smoothing energy efficience %f, %f, %f' % (epoch, smoothingEnergyEfficienceList[-1].max(), smoothingEnergyEfficienceList[-1].min(), smoothingEnergyEfficienceList[-1].mean()))

            stopTime = time.time()
            print(f'last {displayEvery_n_epochs} epochs took {stopTime - startTime} sec')
            startTime = time.time()

            if optimizationMode == 'maximizeFiltering':
                if filteringEnergyEfficienceList[-1].max() > maxFilterEnergyEfficience:
                    maxFilterEnergyEfficience = filteringEnergyEfficienceList[-1].max()
                    print('SAVED epoch: %d - max,min,mean mean filtering energy efficience %f, %f, %f' % (epoch, filteringEnergyEfficienceList[-1].max(), filteringEnergyEfficienceList[-1].min(), filteringEnergyEfficienceList[-1].mean()))
                    pickle.dump({"sysModel": sysModel, "u": u.detach().cpu().numpy(), "uMeanNormList": uMeanNormList, "filteringEnergyEfficienceList": filteringEnergyEfficienceList}, open(filePath, "wb"))
            elif optimizationMode == 'maximizeSmoothing':
                if smoothingEnergyEfficienceList[-1].max() > maxFilterEnergyEfficience:
                    maxFilterEnergyEfficience = smoothingEnergyEfficienceList[-1].max()
                    print('SAVED epoch: %d - max,min,mean mean smoothing energy efficience %f, %f, %f' % (epoch, smoothingEnergyEfficienceList[-1].max(), smoothingEnergyEfficienceList[-1].min(), smoothingEnergyEfficienceList[-1].mean()))
                    pickle.dump({"sysModel": sysModel, "u": u.detach().cpu().numpy(), "uMeanNormList": uMeanNormList, "smoothingEnergyEfficienceList": smoothingEnergyEfficienceList}, open(filePath, "wb"))
        

if enableInvestigation:
    model_results = pickle.load(open(filePath, "rb"))
    sysModel = model_results["sysModel"]
    u = model_results["u"]
    z = np.matmul(np.transpose(sysModel["H"]), u)
    uMeanNorms = np.array(model_results["uMeanNormList"])
    if optimizationMode == 'maximizeFiltering':
        filteringEnergyEfficience = np.array(model_results["filteringEnergyEfficienceList"])
    elif optimizationMode == 'maximizeSmoothing':
        smoothingEnergyEfficience = np.array(model_results["smoothingEnergyEfficienceList"])

    batchSize, dim_x = u.shape[1], sysModel["F"].shape[0]
    # run Anderson's filter & smoother:
    # estimator init values:
    filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
    x_est_f, x_est_s = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
    # x_est_f[k] has the estimation of x[k] given z[0:k-1]
    # x_est_s[k] has the estimation of x[k] given z[0:N-1]
    if optimizationMode == 'maximizeFiltering':
        filteringMeanPowerPerBatch = np.power(x_est_f[1:], 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u[:-1], 2).sum(axis=2).mean(axis=0)
        filteringPowerEfficiencyPerBatch = np.divide(filteringMeanPowerPerBatch, inputMeanPowerPerBatch)
        filteringPowerEfficiencyPerBatch_sortedIndexes = np.argsort(filteringPowerEfficiencyPerBatch, axis=0)
    elif optimizationMode == 'maximizeSmoothing':
        smoothingMeanPowerPerBatch = np.power(x_est_s, 2).sum(axis=2).mean(axis=0)
        inputMeanPowerPerBatch = np.power(u, 2).sum(axis=2).mean(axis=0)
        smoothingPowerEfficiencyPerBatch = np.divide(smoothingMeanPowerPerBatch, inputMeanPowerPerBatch)
        smoothingPowerEfficiencyPerBatch_sortedIndexes = np.argsort(smoothingPowerEfficiencyPerBatch, axis=0)

    for pIdx in range(4):
        if optimizationMode == 'maximizeFiltering':
            maxFilteringPowerEfficiencyIndex = np.argmax(filteringPowerEfficiencyPerBatch[-1-pIdx])
        elif optimizationMode == 'maximizeSmoothing':
            maxFilteringPowerEfficiencyIndex = np.argmax(smoothingPowerEfficiencyPerBatch[-1 - pIdx])

        plt.subplots(nrows=2, ncols=1)
        plt.subplot(2, 1, 1)
        plt.plot(watt2dbm(np.power(uMeanNorms[:, maxFilteringPowerEfficiencyIndex, 0], 2)), label=r'$\frac{1}{N} \sum_{k} {||x^u_k||}_2^2$')
        plt.xlabel('epochs')
        plt.ylabel('dbm')
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        if optimizationMode == 'maximizeFiltering':
            plt.plot(watt2db(filteringEnergyEfficience[:, maxFilteringPowerEfficiencyIndex, 0]), label=r'$\frac{\sum_{k=1}^{N-1} {||\xi_k||}_2^2}{\sum_{k=0}^{N-2} {||x^u_k||}_2^2}$')
        elif optimizationMode == 'maximizeSmoothing':
            plt.plot(watt2db(smoothingEnergyEfficience[:, maxFilteringPowerEfficiencyIndex, 0]), label=r'$\frac{\sum_{k=0}^{N-1} {||\xi^s_k||}_2^2}{\sum_{k=0}^{N-1} {||x^u_k||}_2^2}$')
        plt.xlabel('epochs')
        plt.ylabel('db')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=1)

        enablePlots = False
        if enablePlots:
            plt.subplots(nrows=3, ncols=1)
        else:
            plt.figure()

        if enablePlots: plt.subplot(4, 1, 1)
        u_norm = np.linalg.norm(u[:-1, maxFilteringPowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(u_norm, label=r'${||u||}_2$')
        else:
            plt.plot(u[:-1, maxFilteringPowerEfficiencyIndex, :, 0], label=r'$u$')
        if enablePlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(u_norm, 2).mean())))
            plt.grid()
            plt.legend()

        if enablePlots: plt.subplot(4, 1, 2)
        z_norm = np.linalg.norm(z[:-1, maxFilteringPowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(z_norm, label=r'${||z||}_2$')
        else:
            plt.plot(z[:-1, maxFilteringPowerEfficiencyIndex, :, 0], label=r'$z$')
        if enablePlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||H^{T} x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(z_norm, 2).mean())))
            plt.grid()
            plt.legend()

        if enablePlots: plt.subplot(4, 1, 3)
        x_est_norm = np.linalg.norm(x_est_f[1:, maxFilteringPowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid k-1}||}_2$')
        else:
            plt.plot(x_est_f[:, maxFilteringPowerEfficiencyIndex, :, 0], label=r'$\hat{x}_{k \mid k-1}$')
        if enablePlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||\xi_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(x_est_norm, 2).mean())))
            plt.grid()
            plt.legend()


        if enablePlots: plt.subplot(4, 1, 4)

        x_est_norm = np.linalg.norm(x_est_s[:, maxFilteringPowerEfficiencyIndex, :, 0], axis=1)
        if dim_x > 1:
            plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid N-1}||}_2$')
        else:
            plt.plot(x_est_s[:, maxFilteringPowerEfficiencyIndex, :, 0], label=r'$\hat{x}_{k \mid N-1}$')
        if enablePlots:
            plt.title(r'$\frac{1}{N}\sum_{k} {||\xi^s_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(x_est_norm, 2).mean())))

        plt.grid()
        plt.legend()

        if enablePlots: plt.subplots_adjust(hspace=1)
    plt.show()