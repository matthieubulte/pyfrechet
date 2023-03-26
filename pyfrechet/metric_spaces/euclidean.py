import numpy as np
from .metric_space import MetricSpace

class Euclidean(MetricSpace):
    def __init__(self, dim):
        self.dim = dim
    
    def _d(self, x, y):
        return np.linalg.norm(x - y)
    
    def _frechet_mean(self, y, w):
        return w.dot(y)
    
    def __str__(self):
        return f'Euclidean(dim={self.dim})'
    