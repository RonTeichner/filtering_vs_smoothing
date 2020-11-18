import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *

np.random.seed(11)
dim_x, dim_z = 5, 3
N = 1000  # time steps
batchSize = 16

# estimator init values:
filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.random.randn(dim_x, 1))
# filter_P_init: [1, batchSize, dim_x, dim_x]
# filterStateInit: [1, batchSize, dim_x, 1]


# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)

# create time-series measurements (#time-series == batchSize):
z, _ = GenMeasurements(N, batchSize, sysModel) # z: [N, batchSize, dim_z]

x_est_f, x_est_s = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
# x_est_f[k] has the estimation of x[k] given z[0:k-1]
# x_est_s[k] has the estimation of x[k] given z[0:N-1]

x=3
