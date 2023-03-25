import numpy as np
from scipy.linalg import cho_factor, cho_solve

from metric_spaces import MetricData
from ..weighting_regressor import WeightingRegressor

class GlobalFrechet(WeightingRegressor):
    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.mu = X.mean(axis=0)
        self.centered_x_train = X - self.mu
        self.Sigma = np.cov(X, rowvar=False)
        self.Sigma_chol = cho_factor(self.Sigma)
        return self

    def _weights_for(self, x):
        S_inv_dx = cho_solve(self.Sigma_chol, (x - self.mu).T).T
        return 1 + np.sum(S_inv_dx * self.centered_x_train, axis=1)


    def clone(self):
        return GlobalFrechet()