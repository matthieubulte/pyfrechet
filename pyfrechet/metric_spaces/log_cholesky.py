import numpy as np
from .metric_space import MetricSpace

class LogCholesky(MetricSpace):
    """
    Log-Cholesky space of dxd SPD matrices as defined in [1]. The matrices are represented
    via their Cholesky decomposition by the tuple (D, L) where D are the diagonal elements
    of the Cholesky factor and L is the lower triangular without the diagonal factors.

    Then, for A = (D1, L1) and B = (D2, L2)
    d(A, B)^2 = || L1 - L2 ||_F^2 + || log D1 - log D2 ||_2^2

    For a rv X = (D, L), 
    E[X] = (exp(E[log D], E[L]) 

    We represent this in a vector [ log D; vec(L) ] of dim d(d+1)/2. Transforming from and 
    to the full matrix representation can be done via the spd_to_log_chol and log_chol_to_spd
    functions.

    [1] RIEMANNIAN GEOMETRY OF SYMMETRIC POSITIVE DEFINITE MATRICES VIA CHOLESKY DECOMPOSITION
    """
    def __init__(self, dim):
        self.dim = dim

    def _d(self, x, y):
        return np.linalg.norm(x - y)
    
    def _frechet_mean(self, y, w):
        return w.dot(y)

    def __str__(self):
        return f'LogCholesky({self.dim}x{self.dim})'


def spd_to_log_chol(X):
    d = X.shape[0]
    L = np.linalg.cholesky(X)
    return np.r_[np.log(L[np.diag_indices(d)]), L[np.tril_indices(d, -1)]]


def log_chol_to_spd(DL):
    n = DL.shape[0]
    d = int((-1 + np.sqrt(1 + 8 * n)) / 2)
    L = np.zeros(shape=(d,d))
    L[np.diag_indices(d)] = np.exp(DL[:d])
    L[np.tril_indices(d, -1)] = DL[d:]
    return L.dot(L.T)