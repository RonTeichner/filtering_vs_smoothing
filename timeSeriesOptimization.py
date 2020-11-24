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
filePath = "./minimizeFiltering.pt"

if enableOptimization:
    #np.random.seed(11)
    dim_x, dim_z = 2, 2
    N = 200  # time steps
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
    pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=False, useCuda=useCuda)
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
    optimizerBreakThr = -30 # dbm

    uMeanNormList, uHighestNorm, uMeanNorm, epochIndex, zHighestNorm, zMeanNorm, filteringEnergyList = list(), list(), list(), list(), list(), list(), list()
    epoch = -1
    displayEvery_n_epochs = 100
    startTime = time.time()
    while True:
        epoch += 1
        uMeanNormList.append(np.linalg.norm(u.detach().cpu().numpy(), axis=2).mean(axis=0))
        #print(f'starting epoch {epoch}')
        optimizer.zero_grad()
        z = torch.matmul(H_transpose, u)
        x_est_f, x_est_s = pytorchEstimator(z, filterStateInit)
        filteringEnergy = calcTimeSeriesMeanEnergy(x_est_f)  # mean energy at every batch [volt]
        smoothingEnergy = calcTimeSeriesMeanEnergy(x_est_s)  # mean energy at every batch [volt]
        loss = torch.mean(filteringEnergy)
        # loss.is_contiguous()
        loss.backward()
        optimizer.step()  # parameter update

        filteringEnergyList.append(filteringEnergy)
        if np.mod(epoch, displayEvery_n_epochs) == 0:
            uHighestNorm.append(np.linalg.norm(u.detach().cpu().numpy()[:-1], axis=2).max())
            uMeanNorm.append(np.linalg.norm(u.detach().cpu().numpy()[:-1], axis=2).mean())
            zHighestNorm.append(np.linalg.norm(torch.matmul(H_transpose, u).detach().cpu().numpy()[:-1], axis=2).max())
            zMeanNorm.append(np.linalg.norm(torch.matmul(H_transpose, u).detach().cpu().numpy()[:-1], axis=2).mean())
            epochIndex.append(epoch)
            print('epoch: %d - max,min,mean mean filtering energy %2.2f, %2.2f, %2.2f dbm; uHighestInput: %2.2f dbm; uMeanInput: %2.2f dbm; zHighestInput: %2.2f dbm; zMeanInput: %2.2f dbm' % (epoch, volt2dbm(filteringEnergy.max().item()), volt2dbm(filteringEnergy.min().item()), volt2dbm(filteringEnergy.mean().item()), volt2dbm(uHighestNorm[-1]), volt2dbm(uHighestNorm[-1]), volt2dbm(zHighestNorm[-1]), volt2dbm(zHighestNorm[-1])))

            stopTime = time.time()
            print(f'last {displayEvery_n_epochs} epochs took {stopTime - startTime} sec')
            startTime = time.time()

        if volt2dbm(filteringEnergy.max().item()) < optimizerBreakThr:
            print('Last epoch: %d - max,min,mean mean filtering energy %2.2f, %2.2f, %2.2f dbm; uHighestInput: %2.2f dbm; uMeanInput: %2.2f dbm; zHighestInput: %2.2f dbm; zMeanInput: %2.2f dbm' % (epoch, volt2dbm(filteringEnergy.max().item()), volt2dbm(filteringEnergy.min().item()), volt2dbm(filteringEnergy.mean().item()), volt2dbm(uHighestNorm[-1]), volt2dbm(uHighestNorm[-1]), volt2dbm(zHighestNorm[-1]), volt2dbm(zHighestNorm[-1])))
            pickle.dump({"sysModel": sysModel, "u": u.detach().cpu().numpy(), "uMeanNormList": uMeanNormList}, open(filePath, "wb"))
            break

if enableInvestigation:
    model_results = pickle.load(open(filePath, "rb"))
    sysModel = model_results["sysModel"]
    u = model_results["u"]
    z = np.matmul(np.transpose(sysModel["H"]), u)
    uMeanNorms = np.array(model_results["uMeanNormList"])
    batchSize, dim_x = u.shape[1], sysModel["F"].shape[0]
    # run Anderson's filter & smoother:
    # estimator init values:
    filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
    filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.zeros((dim_x, 1)))
    x_est_f, x_est_s = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
    # x_est_f[k] has the estimation of x[k] given z[0:k-1]
    # x_est_s[k] has the estimation of x[k] given z[0:N-1]
    minNormFilteringIndex = np.linalg.norm(x_est_f, axis=2).mean(axis=0).argmin()

    plt.figure()
    plt.plot(watt2dbm(np.power(uMeanNorms[:, minNormFilteringIndex, 0], 2)), label=r'$\frac{1}{N} \sum_{k} {||x^u_k||}_2^2$')
    plt.xlabel('epochs')
    plt.ylabel('dbm')
    plt.grid()
    plt.legend()

    plt.subplots(nrows=3, ncols=1)

    plt.subplot(3, 1, 1)
    u_norm = np.linalg.norm(u[:-1, minNormFilteringIndex, :, 0], axis=1)
    plt.plot(u_norm, label=r'${||u||}_2$')
    plt.title(r'$\frac{1}{N}\sum_{k} {||x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(u_norm, 2).mean())))
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    z_norm = np.linalg.norm(z[:-1, minNormFilteringIndex, :, 0], axis=1)
    plt.plot(z_norm, label=r'${||z||}_2$')
    plt.title(r'$\frac{1}{N}\sum_{k} {||H^{T} x^u_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(z_norm, 2).mean())))
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    x_est_norm = np.linalg.norm(x_est_f[:, minNormFilteringIndex, :, 0], axis=1)
    plt.plot(x_est_norm, label=r'${||\hat{x}_{k \mid k-1}||}_2$')
    plt.title(r'$\frac{1}{N}\sum_{k} {||\xi_{k}||}_2^2 = $ %2.2f dbm' % (watt2dbm(np.power(x_est_norm, 2).mean())))
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=1)
    plt.show()