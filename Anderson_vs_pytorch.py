import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *

np.random.seed(11)
dim_x, dim_z = 5, 3
N = 1000  # time steps
batchSize = 16

# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)

# create time-series measurements (#time-series == batchSize):
z, _ = GenMeasurements(N, batchSize, sysModel)
x=3
