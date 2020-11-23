import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db, volt2dbm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

#np.random.seed(11)
dim_x, dim_z = 1, 1
N = 10  # time steps
batchSize = 16
num_epochs = 200000
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
pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, useCuda)
if useCuda:
    pytorchEstimator = pytorchEstimator.cuda()
pytorchEstimator.eval()
# create input time-series:
u = torch.randn((N, batchSize, dim_x, 1), dtype=torch.float)
if useCuda:
    u = u.cuda()
#model = time_series_model(N, batchSize, dim_x)

# perform an optimization:
optimizer = optim.SGD([u.requires_grad_()], lr=0.001)
optimizerThr = -30 # dbm

uHighestNorm, uMeanNorm, epochIndex, zHighestNorm, zMeanNorm, filteringEnergyList = list(), list(), list(), list(), list(), list()
epoch = -1
displayEvery_n_epochs = 100
startTime = time.time()
while True:
    epoch += 1
    #print(f'starting epoch {epoch}')
    optimizer.zero_grad()
    x_est_f, x_est_s = pytorchEstimator(torch.matmul(H_transpose, u), filterStateInit)
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

    if volt2dbm(filteringEnergy.max().item()) < -30:
        print('Last epoch: %d - max,min,mean mean filtering energy %2.2f, %2.2f dbm; uHighestInput: %2.2f dbm; uMeanInput: %2.2f dbm; zHighestInput: %2.2f dbm; zMeanInput: %2.2f dbm' % (epoch, volt2dbm(filteringEnergy.max().item()), volt2dbm(filteringEnergy.min().item()), volt2dbm(filteringEnergy.mean().item()), volt2dbm(uHighestNorm[-1]), volt2dbm(uHighestNorm[-1]), volt2dbm(zHighestNorm[-1]), volt2dbm(zHighestNorm[-1])))
        pickle.dump({"sysModel": sysModel, "u": u}, open("minimizeFiltering.pt", "wb"))
        break
