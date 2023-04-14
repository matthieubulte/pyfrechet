import numpy as np
from ..metric_space import MetricSpace
from .nearcorr import nearcorr

class CorrFrobenius(MetricSpace):
    def __init__(self, dim, nearcorr_tol=1e-8):
        self.dim = dim
        self.nearcorr_tol = nearcorr_tol
    
    def _d(self, x, y):
        return np.linalg.norm(x - y, 'fro')
    
    def _frechet_mean(self, y, w):
        B = (w[:,None,None] * y).sum(axis=0)
        return self.project(B)
    
    def project(self, x):
        return nearcorr(x, tol=[self.nearcorr_tol, self.nearcorr_tol])

    def index(self, y, i):
        return y[i, :, :]

    def __str__(self):
        return f'CorrFrobenius(dim={self.dim})'
    