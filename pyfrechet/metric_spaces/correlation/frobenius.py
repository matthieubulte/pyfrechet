import numpy as np
from ..metric_space import MetricSpace
from .nearest_correlation.nearest_correlation import nearcorr

class CorrFroebenius(MetricSpace):
    def __init__(self, dim):
        self.dim = dim
    
    def _d(self, x, y):
        return np.linalg.norm(x - y, 'fro')
    
    def _frechet_mean(self, y, w):
        B = (w[:,None,None] * y).sum(axis=0)
        return nearcorr(B)
    
    def index(self, y, i):
        return y[i, :, :]

    def __str__(self):
        return f'CorrFrobenius(dim={self.dim})'
    