from joblib import Parallel, delayed
import numpy as np

# def D_mat(M, y):
#     N = y.shape[0]
#     D = np.zeros((N,N))
#     for i in range(N):
#         D[i, i+1:] = M.d(M.index(y[[i],:], y[i+1:,])**2
#     return D + D.T

def D_mat(M, y):
    N = y.shape[0]
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            D[i, j] = M.d(M.index(y, i), M.index(y, j))**2
    return D + D.T

def D_mat_par(M, y, n_jobs=-2):
    N = y.shape[0]
    def calc(i, _y): 
        D = np.zeros(N)
        for j in range(i+1,N):
            D[j] = M.d(M.index(_y, i), M.index(_y, j))**2
        return D
    D = Parallel(n_jobs=n_jobs, verbose=1)(delayed(calc)(i, y) for i in range(N))
    D = np.r_[D]
    return D + D.T

def medoid_var(D):
    return np.min(D.sum(axis=1))

def mat_sel_idx(D, idx):
    return D[idx,:][:,idx]

def mat_sel(D, mask):
    return mat_sel_idx(D, np.argwhere(mask)[:,0])

def coalesce_weights(w, shaped) -> np.ndarray:
    N = shaped.shape[0]
    return w if not w is None else np.ones(N)/N