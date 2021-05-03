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

enablePlotOnly = False
enableInvestigateAllN = True
fileName = 'sys2D_secondTry'

if enablePlotOnly:
    savedList = pickle.load(open(fileName + '_final_' + '.pt', "rb"))
    sysModel, bounds_N, currentFileName_N, bounds_N_plus_m, currentFileName_N_plus_m, bounds_N_plus_2m, currentFileName_N_plus_2m, mistakeBound, delta_trS, gapFromInfBound = savedList
    savedList = pickle.load(open(currentFileName_N_plus_2m, "rb"))
    N = savedList[2]

    print(f'bounds file loaded; bounds were calculated for N = {N}')
    print(f'no player bound is {watt2dbm(bounds_N_plus_2m[0])} dbm')
    print(f'no knowledge bound is {watt2dbm(bounds_N_plus_2m[1])} dbm; {watt2dbm(bounds_N_plus_2m[1]) - watt2dbm(bounds_N_plus_2m[0])} db')
    print(f'no access bound is {watt2dbm(bounds_N_plus_2m[2])} dbm; {watt2dbm(bounds_N_plus_2m[2]) - watt2dbm(bounds_N_plus_2m[0])} db')
    print(f'causal bound is {watt2dbm(bounds_N_plus_2m[3])} dbm; {watt2dbm(bounds_N_plus_2m[3]) - watt2dbm(bounds_N_plus_2m[0])} db')
    print(f'genie bound is {watt2dbm(bounds_N_plus_2m[4])} dbm; {watt2dbm(bounds_N_plus_2m[4]) - watt2dbm(bounds_N_plus_2m[0])} db')

    # plotting:
    adversarialPlayerPlotting(currentFileName_N_plus_2m)
    plt.show()
    exit()

if enableInvestigateAllN:
    useCuda = False
    mistakeBound, delta_trS = 1 * 1e-2, 1 * 1e-2
    enableCausalPlayer = True
    savedList = pickle.load(open(fileName + '_final_' + '.pt', "rb"))
    sysModel, _, _, _, _, _, _, _, _, _ = savedList
    fileName = fileName + '_allN_'
    pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)
    delta_u, delta_caligraphE = 1e-3, 1e-3
    adversarialPlayersToolbox = playersToolbox(pytorchEstimator, delta_u, delta_caligraphE, True)
    N_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,40,50,60,70,80,90,100]
    bounds_N_list = list()
    for N in N_list:
        print(f'all N calculations, N = {N}')
        bounds_N, _ = runBoundSimulation(sysModel, pytorchEstimator, adversarialPlayersToolbox, useCuda, True, N, mistakeBound, delta_trS, enableCausalPlayer, fileName)
        bounds_N_list.append(bounds_N_list)
    pickle.dump([sysModel, bounds_N_list], open(fileName + '.pt', 'wb'))


dim_x, dim_z = 2, 2
seed = 13

useCuda = False

initialN = 50
factorN = 1.5
gapFromInfBound = 5*1e-2  # w.r.t tr{Sigma}

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

trS = np.trace(Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=False).theoreticalBarSigma.cpu().numpy())
gap_wrt_trSigma = gapFromInfBound * trS
enableCausalPlayer = False  # fastest run

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)
if useCuda:
    pytorchEstimator = pytorchEstimator.cuda()
pytorchEstimator.eval()

delta_u, delta_caligraphE = 1e-3, 1e-3
adversarialPlayersToolbox = playersToolbox(pytorchEstimator, delta_u, delta_caligraphE, True)
usePreviousRoundResults = False
while True:
    if usePreviousRoundResults:
        bounds_N, currentFileName_N = bounds_N_plus_m, currentFileName_N_plus_m
    else:
        bounds_N, currentFileName_N = runBoundSimulation(sysModel, pytorchEstimator, adversarialPlayersToolbox, useCuda, True, N, mistakeBound, delta_trS, enableCausalPlayer, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N)

    if not usePreviousRoundResults:
        m = int(np.ceil(factorN*N)) - N  # time steps

    if usePreviousRoundResults:
        bounds_N_plus_m, currentFileName_N_plus_m = bounds_N_plus_2m, currentFileName_N_plus_2m
    else:
        bounds_N_plus_m, currentFileName_N_plus_m = runBoundSimulation(sysModel, pytorchEstimator, adversarialPlayersToolbox,  useCuda, True, N + m, mistakeBound, delta_trS, enableCausalPlayer, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N_plus_m)

    bounds_N_plus_2m, currentFileName_N_plus_2m = runBoundSimulation(sysModel, pytorchEstimator, adversarialPlayersToolbox,  useCuda, True, N + 2*m, mistakeBound, delta_trS, enableCausalPlayer, fileName)
    # plotting:
    # adversarialPlayerPlotting(currentFileName_N_plus_2m)
    # plt.show()

    deltaBounds_N = np.subtract(bounds_N_plus_m, bounds_N)
    deltaBounds_N_plus_m = np.subtract(bounds_N_plus_2m, bounds_N_plus_m)

    alpha_N_m = np.divide(deltaBounds_N_plus_m, deltaBounds_N)

    boundsAtInf = bounds_N + np.divide(deltaBounds_N, 1 - alpha_N_m)
    gapMax = np.max(np.abs(np.subtract(boundsAtInf, bounds_N)))

    print(f'max gap from inf is {watt2dbm(gapMax) - watt2dbm(trS)} db w.r.t tr(Sigma)')
    if gapMax > gap_wrt_trSigma:
        N = N + m
        usePreviousRoundResults = True
    else:
        if not enableCausalPlayer:
            enableCausalPlayer = True
            usePreviousRoundResults = False
            continue

        pickle.dump([sysModel, bounds_N, currentFileName_N, bounds_N_plus_m, currentFileName_N_plus_m, bounds_N_plus_2m, currentFileName_N_plus_2m, mistakeBound, delta_trS, gapFromInfBound],
                    open(fileName + '_final_' + '.pt', 'wb'))
        print('bounds file saved')
        print(f'no player bound is {watt2dbm(bounds_N[0])} dbm')
        print(f'no knowledge bound is {watt2dbm(bounds_N[1])} dbm')
        print(f'no access bound is {watt2dbm(bounds_N[2])} dbm')
        print(f'causal bound is {watt2dbm(bounds_N[3])} dbm')
        print(f'genie bound is {watt2dbm(bounds_N[4])} dbm')
        # plotting:
        adversarialPlayerPlotting(currentFileName_N_plus_2m)
        plt.show()
        break

