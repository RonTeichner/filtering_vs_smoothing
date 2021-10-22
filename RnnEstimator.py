import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pytorchKalman_func import *
from analyticResults_func import watt2db, volt2dbm, watt2dbm, calc_tildeC
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time

fileName = 'sys2D_FilteringVsSmoothing'
savedGammaResults = pickle.load(open(fileName + '_gammaResults.pt', 'rb'))
sysModel, N, gammaResultList = savedGammaResults

# training properties:
batchSize = 8*40
validation_fraction = 0.2
nSeriesForTrain = 10000
nSeriesForTest = 10000

class MeasurementsDataset(Dataset):
    def __init__(self, sysModel, nTime, nSeries, transform=None):

        self.completeDataSet = self.getBatch(sysModel, nTime, nSeries)

        self.transform = transform

    def getBatch(self, sysModel, nTimePoints, nSeries):
        device = 'cpu'
        tilde_z, tilde_x, processNoises, measurementNoises = GenMeasurements(nTimePoints, nSeries, sysModel, startAtZero=False, dp=False)  # z: [N, nSeries, dim_z]

        # unmodeled behavior:
        tr_Q = np.trace(sysModel['Q'])
        u = np.zeros_like(tilde_x)
        #u[:, :, 0, 0] = np.power(tilde_x[:, :, 1, 0], 1)
        u[:, :, 0, 0] = np.power(tilde_x[:, :, 1, 0], 2) * (tilde_x[:, :, 1, 0] > 0)
        u[:, :, 1, 0] = np.power(tilde_x[:, :, 0, 0], 2) * (tilde_x[:, :, 0, 0] > 0)
        #u_mean_watt = np.power(u, 2).sum(axis=2).flatten().mean()
        #alpha = 0
        #u = alpha*u
        u_mean_watt = np.power(u, 2).sum(axis=2).flatten().mean()
        gamma_wrt_tr_Q = u_mean_watt/tr_Q
        print(f'$\gamma/tr(Q)$ = {gamma_wrt_tr_Q}')
        np.power(np.matmul(np.transpose(sysModel['H']), u), 2).sum(axis=2).flatten().mean()

        z = tilde_z + np.matmul(np.transpose(sysModel['H']), u)

        tilde_z, tilde_x, processNoises, measurementNoises = torch.tensor(tilde_z, dtype=torch.float, device=device), torch.tensor(tilde_x, dtype=torch.float, device=device), torch.tensor(processNoises, dtype=torch.float, device=device), torch.tensor(measurementNoises, dtype=torch.float, device=device)
        z = torch.tensor(z, dtype=torch.float, device=device)

        return {'z': z, 'tilde_z': tilde_z, 'tilde_x': tilde_x}

    def __len__(self):
        return self.completeDataSet['tilde_z'].shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'z': self.completeDataSet['z'][:, idx], 'tilde_z': self.completeDataSet['tilde_z'][:, idx], 'x': self.completeDataSet['tilde_x'][:, idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# class definition
class RNN_Filter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN_Filter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # setup RNN layer
        #self.Adv_rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers)
        self.Filter_rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, z):
        # z_k of shape: [N, batchSize, dim_z]
        controlHiddenDim, hidden = self.Filter_rnn(z[:, :, :, 0])
        # controlHiddenDim shape: [N, batchSize, hidden_dim]
        hat_x_k_plus_1_given_k = self.linear(controlHiddenDim)

        return hat_x_k_plus_1_given_k

def trainModel(model, pytorchEstimator, trainLoader, validationLoader):
    filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
    criterion = nn.MSELoss()
    lowThrLr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, threshold=1e-6)
    # moving model to cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    x_0_train_given_minus_1 = torch.zeros((1, trainLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
    x_0_validation_given_minus_1 = torch.zeros((1, validationLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
    # training and saving the model when validation is best:
    print('start training')
    min_valid_loss = np.inf
    epoch = -1
    while True:
        epoch += 1
        train_loss = 0.0
        model.train()

        for i_batch, sample_batched in enumerate(trainLoader):
            # print(f'starting epoch {epoch}, batch {i_batch}')
            optimizer.zero_grad()

            z = sample_batched["z"].transpose(1, 0)
            z = z.to(device)
            currentBatchSize = z.shape[1]
            hat_x_k_plus_1_given_k = model(z)
            learned_tilde_x_est_f = torch.cat((x_0_train_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

            # estimator init values:
            filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
            filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
            # filterStateInit = tilde_x[0]  This can be used if the initial state is known

            # kalman filter on z:
            tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)
            loss = criterion(learned_tilde_x_est_f, tilde_x_est_s)

            loss.backward()
            optimizer.step()  # parameter update

            train_loss += loss.item()

        scheduler.step(train_loss)

        validation_loss = 0.0
        model.eval()
        for i_batch, sample_batched in enumerate(validationLoader):
            z = sample_batched["z"].transpose(1, 0)
            z = z.to(device)
            currentBatchSize = z.shape[1]
            hat_x_k_plus_1_given_k = model(z)
            learned_tilde_x_est_f = torch.cat((x_0_validation_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

            # estimator init values:
            filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
            filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
            # filterStateInit = tilde_x[0]  This can be used if the initial state is known

            # kalman filter on z:
            tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)
            loss = criterion(learned_tilde_x_est_f, tilde_x_est_s)
            validation_loss += loss.item()

        validation_loss = validation_loss/(i_batch+1)
        if min_valid_loss > validation_loss:
            print(f'epoch {epoch}, Validation loss Decreased({min_valid_loss:.6f}--->{validation_loss:.6f}); lr: {scheduler._last_lr[-1]}')
            min_valid_loss = validation_loss
            torch.save(model.module.state_dict(), 'saved_modelFilter.pt')

        if scheduler._last_lr[-1] < lowThrLr:
            print(f'Stoping optimization due to learning rate of {scheduler._last_lr[-1]}')
            break



# create dataset and split into train and test sets:
print(f'creating dataset')
patientsTrainDataset = MeasurementsDataset(sysModel, N, nSeriesForTrain)

nValidation = int(validation_fraction*len(patientsTrainDataset))
nTrain = len(patientsTrainDataset) - nValidation
trainData, validationData = random_split(patientsTrainDataset,[nTrain, nValidation])
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0)
validationLoader = DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=0)

# patient Id's in trainLoader are obtained by, trainLoader.dataset.indices or trainData.indices
useCuda = True
if useCuda:
    device = 'cuda'
else:
    device = 'cpu'

pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=useCuda)

# create the trained model
hidden_dim = 10
num_layers = 2
Filter_rnn = RNN_Filter(input_dim = pytorchEstimator.dim_z, hidden_dim=hidden_dim, output_dim=pytorchEstimator.dim_x, num_layers=num_layers)

trainModel(Filter_rnn, pytorchEstimator, trainLoader, validationLoader)

# test
print('starting test')
pytorchEstimator = Pytorch_filter_smoother_Obj(sysModel, enableSmoothing=True, useCuda=False)
Filter_rnn = RNN_Filter(input_dim = pytorchEstimator.dim_z, hidden_dim=hidden_dim, output_dim=pytorchEstimator.dim_x, num_layers=num_layers)
device = 'cpu'
Filter_rnn.load_state_dict(torch.load('saved_modelFilter.pt'))
Filter_rnn.eval().to(device)
patientsTestDataset = MeasurementsDataset(sysModel, N, nSeriesForTest)
testLoader = DataLoader(patientsTestDataset, batch_size=nSeriesForTest, shuffle=True, num_workers=0)
x_0_test_given_minus_1 = torch.zeros((1, testLoader.batch_size, pytorchEstimator.dim_x), dtype=torch.float, device=device)
filter_P_init = pytorchEstimator.theoreticalBarSigma.cpu().numpy()  # filter @ time-series but all filters have the same init
for i_batch, sample_batched in enumerate(testLoader):
    tilde_z = sample_batched["tilde_z"].transpose(1, 0)
    z = sample_batched["z"].transpose(1, 0)
    x = sample_batched["x"].transpose(1, 0)

    currentBatchSize = z.shape[1]
    hat_x_k_plus_1_given_k = Filter_rnn(z)
    learned_tilde_x_est_f = torch.cat((x_0_test_given_minus_1[:, :currentBatchSize], hat_x_k_plus_1_given_k[:-1]), dim=0)[:, :, :, None]

    # estimator init values:
    filterStateInit = np.matmul(np.linalg.cholesky(filter_P_init), np.random.randn(currentBatchSize, pytorchEstimator.dim_x, 1))
    filterStateInit = torch.tensor(filterStateInit, dtype=torch.float, requires_grad=False, device=device).contiguous()
    # filterStateInit = tilde_x[0]  This can be used if the initial state is known

    # kalman filter on z:
    tilde_x_est_f, tilde_x_est_s = pytorchEstimator(z, filterStateInit)

    learned_e_k_give_k_minus_1 = (x - learned_tilde_x_est_f).detach().numpy()
    kalman_e_k_give_k_minus_1 = (x - tilde_x_est_f).detach().numpy()
    kalman_e_k_give_N_minus_1 = (x - tilde_x_est_s).detach().numpy()

    learned_e_k_give_k_minus_1_watt = np.power(learned_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
    kalman_e_k_give_k_minus_1_watt = np.power(kalman_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
    kalman_e_k_give_N_minus_1_watt = np.power(kalman_e_k_give_N_minus_1, 2).sum(axis=2).flatten()

    # kalman filter on tilde_z:
    pure_tilde_x_est_f, pure_tilde_x_est_s = pytorchEstimator(tilde_z, filterStateInit)

    pure_kalman_e_k_give_k_minus_1 = (x - pure_tilde_x_est_f).detach().numpy()
    pure_kalman_e_k_give_N_minus_1 = (x - pure_tilde_x_est_s).detach().numpy()

    pure_kalman_e_k_give_k_minus_1_watt = np.power(pure_kalman_e_k_give_k_minus_1, 2).sum(axis=2).flatten()
    pure_kalman_e_k_give_N_minus_1_watt = np.power(pure_kalman_e_k_give_N_minus_1, 2).sum(axis=2).flatten()

    print(f'pure kalman filter MSE {watt2dbm(pure_kalman_e_k_give_k_minus_1_watt.mean())} dbm; {(pure_kalman_e_k_give_k_minus_1_watt.mean())} W')
    print(f'kalman filter MSE {watt2dbm(kalman_e_k_give_k_minus_1_watt.mean())} dbm; {(kalman_e_k_give_k_minus_1_watt.mean())} W')
    print(f'learned filter MSE {watt2dbm(learned_e_k_give_k_minus_1_watt.mean())} dbm; {(learned_e_k_give_k_minus_1_watt.mean())} W')

    print(f'pure kalman smoother MSE {watt2dbm(pure_kalman_e_k_give_N_minus_1_watt.mean())} dbm; {(pure_kalman_e_k_give_N_minus_1_watt.mean())} W')
    print(f'kalman smoother MSE {watt2dbm(kalman_e_k_give_N_minus_1_watt.mean())} dbm; {(kalman_e_k_give_N_minus_1_watt.mean())} W')

    totalKalmanIncrease = kalman_e_k_give_k_minus_1_watt.mean() - pure_kalman_e_k_give_k_minus_1_watt.mean()
    print(f'kalman filter error increases by {totalKalmanIncrease} [W] due to the unmodeled behavior')
    decreaseOfLearned = kalman_e_k_give_k_minus_1_watt.mean() - learned_e_k_give_k_minus_1_watt.mean()
    print(f'learned filter error is {decreaseOfLearned} [W] below the standart kalman filter')
    print(f'learned filter removed {decreaseOfLearned/totalKalmanIncrease*100} % of the increase in error of the standart filter')



    plt.figure()
    n_bins = 1000
    n, bins, patches = plt.hist(watt2dbm(pure_kalman_e_k_give_k_minus_1_watt), n_bins, color='green', linestyle = 'dashed', density=True, histtype='step', cumulative=True)
    n, bins, patches = plt.hist(watt2dbm(pure_kalman_e_k_give_N_minus_1_watt), n_bins, color='blue', linestyle = 'dashed', density=True, histtype='step', cumulative=True)

    n, bins, patches = plt.hist(watt2dbm(kalman_e_k_give_k_minus_1_watt), n_bins, color='green', density=True, histtype='step', cumulative=True, label=r'Kalman filter')
    n, bins, patches = plt.hist(watt2dbm(kalman_e_k_give_N_minus_1_watt), n_bins, color='blue', density=True, histtype='step', cumulative=True, label=r'Kalman smoother')
    n, bins, patches = plt.hist(watt2dbm(learned_e_k_give_k_minus_1_watt), n_bins, color='orange', density=True, histtype='step', cumulative=True, label=r'learned filter')
    plt.xlabel('dbm')
    plt.title(r'CDF of estimation errors')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
