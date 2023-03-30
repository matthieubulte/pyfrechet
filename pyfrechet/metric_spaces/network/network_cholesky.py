import numpy as np
from ..metric_space import MetricSpace

class NetworkCholesky(MetricSpace):
    def __init__(self, dim):
        self.dim = dim
    
    def _d(self, x, y):
        raise NotImplementedError()
    
    def _frechet_mean(self, y, w):
        raise NotImplementedError()
    
    def index(self, y, i):
        return y[i, :, :]

    def __str__(self):
        return f'NetworkCholesky(dim={self.dim})'
    