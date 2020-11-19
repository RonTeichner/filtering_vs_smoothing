import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db

np.random.seed(11)
dim_x, dim_z = 5, 3
N = 1000  # time steps
batchSize = 1

# estimator init values:
filter_P_init = np.repeat(np.eye(dim_x)[None, None, :, :], batchSize, axis=1)  # filter @ time-series but all filters have the same init
filterStateInit = np.dot(np.linalg.cholesky(filter_P_init), np.random.randn(dim_x, 1))
# filter_P_init: [1, batchSize, dim_x, dim_x]
# filterStateInit: [1, batchSize, dim_x, 1]


# create a single system model:
sysModel = GenSysModel(dim_x, dim_z)

# create time-series measurements (#time-series == batchSize):
z, _ = GenMeasurements(N, batchSize, sysModel) # z: [N, batchSize, dim_z]

# run Anderson's filter & smoother:
x_est_f_A, x_est_s_A = Anderson_filter_smoother(z, sysModel, filter_P_init, filterStateInit)
# x_est_f[k] has the estimation of x[k] given z[0:k-1]
# x_est_s[k] has the estimation of x[k] given z[0:N-1]

# run pytorch filter & smoother:
# (filter_P_init is not in use because this filter works from the start on the steady-state-gain)
x_est_f_P, x_est_s_P = Pytorch_filter_smoother(z, sysModel, filterStateInit)

x_est_f_P = x_est_f_P.cpu().numpy()
x_est_s_P = x_est_s_P.cpu().numpy()

filtering_recursiveAnderson_recursivePytorch_diff_energy = watt2db(np.power(np.linalg.norm(x_est_f_A[:, 0] - x_est_f_P[:, 0], axis=1), 2))
smoothing_recursiveAnderson_recursivePytorch_diff_energy = watt2db(np.power(np.linalg.norm(x_est_s_A[:, 0] - x_est_s_P[:, 0], axis=1), 2))

plt.figure(figsize=(16, 8))
plt.plot(filtering_recursiveAnderson_recursivePytorch_diff_energy, label='Filtering')
plt.plot(smoothing_recursiveAnderson_recursivePytorch_diff_energy, label='Smoothing')
plt.title(r'Anderson vs Pytorch')
plt.ylabel('db')
plt.legend()
plt.grid()
plt.show()
