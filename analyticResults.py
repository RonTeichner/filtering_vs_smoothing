import numpy as np
import matplotlib.pyplot as plt

def calc_sigma_bar(H, R, F, Q):
    F_t = F.transpose()
    H_t = H.transpose()
    sigma_bar = np.eye(2)
    nIter = 0
    factor = 1
    while True:
        nIter += 1
        if nIter == 100000:
            factor = 20000
            print('calc_sigma_bar: factor changes')
        a = np.matmul(H_t, np.matmul(sigma_bar, H)) + R
        a_inv = np.linalg.inv(a)
        b = np.matmul(sigma_bar, np.matmul(H, np.matmul(a_inv, np.matmul(H_t, sigma_bar))))
        sigma_bar_new = np.matmul(F, np.matmul(sigma_bar - b, F_t)) + Q
        sigma_bar_ratio = np.sum(np.abs(sigma_bar_new / sigma_bar)) / F.size
        sigma_bar = sigma_bar_new
        if np.abs(sigma_bar_ratio - 1) <= factor * np.finfo(np.float).resolution:
            break
    return sigma_bar


def calc_sigma_smoothing(sigma_bar, H, F, R):
    H_t = H.transpose()
    inv_mat = np.linalg.inv(np.matmul(H_t, np.matmul(sigma_bar, H)) + R)
    K = np.matmul(F, np.matmul(sigma_bar, np.matmul(H, inv_mat)))
    F_tilde = F - np.matmul(K, H_t)
    F_tilde_t = F_tilde.transpose()
    a = np.matmul(H_t, np.matmul(sigma_bar, H)) + R
    a_inv = np.linalg.inv(a)
    core = np.matmul(H, np.matmul(a_inv, H_t))
    # i==j:
    summand = 0
    # start with s=k-j=(100-1)
    sInitRange = 100
    for s in range(sInitRange):
        summand += np.matmul(np.linalg.matrix_power(F_tilde_t, s), np.matmul(core, np.linalg.matrix_power(F_tilde, s)))
    sigma_j_k = sigma_bar - np.matmul(sigma_bar, np.matmul(summand, sigma_bar))
    # continue while sigma_j_k changes:
    s = sInitRange
    nIter = 0
    factor = 1
    while True:
        nIter += 1
        if nIter == 100000:
            factor = 20000
            print('calc_sigma_smoothing: factor changes')
        summand += np.matmul(np.linalg.matrix_power(F_tilde_t, s), np.matmul(core, np.linalg.matrix_power(F_tilde, s)))
        sigma_j_k_new = sigma_bar - np.matmul(sigma_bar, np.matmul(summand, sigma_bar))
        sigma_j_k_ratio = np.sum(np.abs(sigma_j_k_new / sigma_j_k)) / F.size
        sigma_j_k = sigma_j_k_new
        s += 1
        if np.abs(sigma_j_k_ratio - 1) <= factor * np.finfo(np.float).resolution:
            break
    return sigma_j_k


beta = 0.1  # acceleration memory
alpha = 0.9  # velocity decay
dt = 0.5  # sec

F = np.array([[beta, 0], [dt, alpha]])
F_t = F.transpose()
H = np.array([[1], [0]])
#H = np.array([[0], [1]])
H_t = H.transpose()
#G = np.eye(2)

std_process_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
std_meas_noises = np.logspace(np.log10(1e-4), np.log10(1e-2), 100, base=10.0)  # np.arange(1e-4, 20e-3, 1e-4)
deltaFS, E_filtering, E_smoothing = np.zeros((std_meas_noises.size, std_process_noises.size)), np.zeros((std_meas_noises.size, std_process_noises.size)), np.zeros((std_meas_noises.size, std_process_noises.size))
i = 0
for pIdx, std_process_noise in enumerate(std_process_noises):
    for mIdx, std_meas_noise in enumerate(std_meas_noises):
        i += 1
        Q = np.array([[np.power(std_process_noise, 2), 0], [0, 0]])
        R = np.power(std_meas_noise, 2)

        #  print(f'eigenvalues of F are: {np.linalg.eig(F)[0]}')

        sigma_bar = calc_sigma_bar(H, R, F, Q)
        sigma_j_k = calc_sigma_smoothing(sigma_bar, H, F, R)

        E_f = np.trace(sigma_bar)
        E_s = np.trace(sigma_j_k)

        deltaFS[mIdx, pIdx] = (E_f - E_s) / (0.5*(E_f + E_s))
        E_filtering[mIdx, pIdx], E_smoothing[mIdx, pIdx] = E_f, E_s
        #  print(f'deltaFS[pIdx, mIdx] = {deltaFS[pIdx, mIdx]}')
    print(f'finished: {100*i/(std_process_noises.size * std_meas_noises.size)} %')


std_process_noises_db = 20*np.log10(std_process_noises/std_process_noises[0])
std_meas_noises_db = 20*np.log10(std_meas_noises/std_meas_noises[0])
X, Y = np.meshgrid(std_process_noises_db, std_meas_noises_db)
Z = deltaFS
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_ylabel('meas noise [db]')
ax.set_xlabel('process noise [db]')
ax.set_title(r'$\frac{tr(\Sigma^F)-tr(\Sigma^S)}{0.5(tr(\Sigma^F)+tr(\Sigma^S))}$')
#plt.show()

Z = 10*np.log10(E_filtering)
Z = Z - Z.max()
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_ylabel('meas noise [db]')
ax.set_xlabel('process noise [db]')
ax.set_title(r'$tr(\Sigma^F)$ [db]')
#plt.show()

Z = 10*np.log10(E_smoothing)
Z = Z - Z.max()
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_ylabel('meas noise [db]')
ax.set_xlabel('process noise [db]')
ax.set_title(r'$tr(\Sigma^S)$ [db]')
plt.show()
'''
n_bins = 50
n, bins, patches = plt.hist(E_filtering.flatten(), n_bins, density=True, histtype='step', cumulative=False, label='Ef')
n, bins, patches = plt.hist(E_smoothing.flatten(), n_bins, density=True, histtype='step', cumulative=False, label='Es')
plt.title('Filtering & Smoothing hist')
plt.xlabel('trace values')
plt.legend()
plt.grid(True)
plt.show()
'''