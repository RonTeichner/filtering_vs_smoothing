import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db, volt2dbm, watt2dbm, calc_tildeC
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

fileName = 'sys1D'
dim_x, dim_z = 1, 1
seed = 13

useCuda = False

initialN = 50
factorN = 1.5
gapFromInfBound = 1*1e-2  # w.r.t tr{Sigma}

mistakeBound, delta_trS = 1*1e-2, 1*1e-2
# P(|bound - estBound| > gamma * Sigma_N) < gamma^{-2} for some gamma > 0
# I want my error on estimating the bound to be w.r.t tr{Sigma}:
# I want the probability of mistaking in more than 1% of tr{Sigma} to be less than 1%
# So gamma * boundVar/M = delta_trS * tr{Sigma} with delta_trS = 0.01 and M the batchSize
# and gamma^{-2} = mistakeBound with mistakeBound = 0.01
# therefore gamma = sqrt(1/mistakeBound)
# and M = (gamma * boundVar) / (delta_trS * tr{Sigma}) = (sqrt(1/mistakeBound) * boundVar) / (delta_trS * tr{Sigma})

np.random.seed(seed)  #  for 2D systems, seed=13 gives two control angles, seed=10 gives multiple angles, seed=9 gives a single angle

# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)

# calc bound on initialN:
N = initialN

gap_wrt_trSigma = gapFromInfBound * np.trace(Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=False).theoreticalBarSigma.cpu().numpy())

while True:
    bounds_N, currentFileName_N = runBoundSimulation(sysModel, useCuda, True, N, mistakeBound, delta_trS, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N)

    m = int(np.ceil(factorN*N)) - N  # time steps
    bounds_N_plus_m, currentFileName_N_plus_m = runBoundSimulation(sysModel, useCuda, True, N + m, mistakeBound, delta_trS, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N_plus_m)

    bounds_N_plus_2m, currentFileName_N_plus_2m = runBoundSimulation(sysModel, useCuda, True, N + 2*m, mistakeBound, delta_trS, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N_plus_2m)
    # plt.show()

    deltaBounds_N = np.subtract(bounds_N_plus_m, bounds_N)
    deltaBounds_N_plus_m = np.subtract(bounds_N_plus_2m, bounds_N_plus_m)

    alpha_N_m = np.divide(deltaBounds_N_plus_m, deltaBounds_N)

    boundsAtInf = bounds_N + np.divide(deltaBounds_N, 1 - alpha_N_m)
    gapMax = np.max(np.abs(np.subtract(boundsAtInf, bounds_N)))

    print(f'max gap from inf is {watt2dbm(gapMax)} dbm')
    if gapMax > gap_wrt_trSigma:
        N = N + 3 * m
    else:
        pickle.dump([sysModel, bounds_N, currentFileName_N, mistakeBound, delta_trS, gapFromInfBound],
                    open(fileName + '_final_' + '.pt', 'wb'))
        print('bounds file saved')
        print(f'no player bound is {watt2dbm(bounds_N[0])} dbm')
        print(f'no knowledge bound is {watt2dbm(bounds_N[1])} dbm')
        print(f'no access bound is {watt2dbm(bounds_N[2])} dbm')
        print(f'causal bound is {watt2dbm(bounds_N[3])} dbm')
        print(f'genie bound is {watt2dbm(bounds_N[4])} dbm')
        # plotting:
        adversarialPlayerPlotting(currentFileName_N)
        plt.show()
        break

