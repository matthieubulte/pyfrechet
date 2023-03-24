from joblib import Parallel, delayed
import numpy as np

def D_mat(d, y):
    N = y.shape[0]
    D = np.zeros((N,N))
    for i in range(N):
        D[i, i+1:] = d(y[[i],:], y[i+1:,])**2
    return D + D.T

def D_mat_sq(d, y):
    N = y.shape[0]
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            D[i, j] = d(y[i], y[j])**2
    return D + D.T

def D_mat_sq_par(d, y, n_jobs=-2):
    N = y.shape[0]
    def calc(i, _y): 
        D = np.zeros(N)
        for j in range(i+1,N):
            D[j] = d(_y[i], _y[j])**2
        return D
    D = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calc)(i, y) for i in range(N))
    D = np.r_[D]
    return D + D.T

def medoid_var(D):
    return np.min(D.sum(axis=1))

def mat_sel_idx(D, idx):
    mask = np.zeros(D.shape[0])
    mask[idx] = 1
    return mat_sel(D, mask)

def mat_sel(D, mask):
    sel = np.argwhere(mask)[:,0]
    return D[sel,:][:,sel]