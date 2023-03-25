import numpy as np
from .weighting_regressor import WeightingRegressor
from metric_spaces import MetricData

class NadarayaWatson(WeightingRegressor):
    def __init__(self, base_kernel, bw=1.0):
        self.base_kernel = base_kernel
        self.bw = bw
        self.x_train = None

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.x_train = X
        return self

    def _weights_for(self, x):
        return self.base_kernel(np.linalg.norm(x - self.x_train, axis=1) / self.bw)
    
    def clone(self):
        return NadarayaWatson(self.base_kernel, self.bw)