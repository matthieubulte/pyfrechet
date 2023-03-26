import numpy as np
from scipy.linalg import cho_factor, cho_solve

from metric_spaces import MetricData
from ..weighting_regressor import WeightingRegressor

class LocalFrechet(WeightingRegressor):
    def __init__(self, base_kernel, bw):
        self.base_kernel = base_kernel
        self.bw = bw

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.x_train = X
        return self

    def weights_for(self, x):
        N = self.x_train.shape[0]
        dx = x - self.x_train
        ks = self.base_kernel(np.linalg.norm(dx, axis=1) / self.bw)
        
        mu1 = ks.dot(dx) / N
        mu2 = (ks[:, None] * dx).T.dot(dx) / N
        mu2_chol = cho_factor(mu2)

        weights = np.sum(ks[:, None] * (1 - mu1) * cho_solve(mu2_chol, dx.T).T, axis=1)
        return self._normalize_weights(weights, sum_to_one=True, clip=True, clip_allow_neg=True)

    def clone(self):
        return LocalFrechet(self.base_kernel, self.bw)