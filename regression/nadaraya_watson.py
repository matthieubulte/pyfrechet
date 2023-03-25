import numpy as np
from .weighting_regressor import WeightingRegressor
from metric_spaces import MetricData

class NadarayaWatson(WeightingRegressor):
    def __init__(self, base_kernel, bw=1.0):
        self.base_kernel = base_kernel
        self.bw = bw
        self.X = None

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.X = X
        return self

    def weights_for(self, x):
        w = self.base_kernel(np.linalg.norm(x - self.X, axis=1) / self.bw)
        return w / w.sum()
    
    def clone(self):
        return NadarayaWatson(self.base_kernel, self.bw)

def epanechnikov(u):
    return 3.0 * (1.0 - np.minimum(1, u**2))/4.0

def gaussian(u):
    return np.exp(-u**2/2)