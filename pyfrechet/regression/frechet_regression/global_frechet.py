import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pyfrechet.metric_spaces import MetricData
from pyfrechet.regression.weighting_regressor import WeightingRegressor

class GlobalFrechet(WeightingRegressor):
    def __init__(self):
        super().__init__(precompute_distances=False)
        
    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.mu_ = X.mean(axis=0)
        self.centered_x_train_ = X - self.mu_
        self.Sigma_ = np.cov(X, rowvar=False)
        self.Sigma_chol_ = cho_factor(self.Sigma_)
        return self

    def weights_for(self, x):
        S_inv_dx = cho_solve(self.Sigma_chol_, (x - self.mu_).T).T
        return self._normalize_weights(1 + np.sum(S_inv_dx * self.centered_x_train_, axis=1), sum_to_one=True, clip=True)
