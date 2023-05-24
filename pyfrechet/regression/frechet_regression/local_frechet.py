import numpy as np

from pyfrechet.metric_spaces import MetricData
from ..weighting_regressor import WeightingRegressor
from ..kernels import gaussian

class LocalFrechet(WeightingRegressor):
    def __init__(self, base_kernel=gaussian, bw=1.0):
        super().__init__(precompute_distances=False)
        self.base_kernel = base_kernel
        self.bw = bw

    def fit(self, X, y: MetricData):
        assert X.shape[1] == 1 # See weights_for

        super().fit(X, y)
        self.x_train_ = X
        return self

    def weights_for(self, x):
        N = self.x_train_.shape[0]
        dx = self.x_train_ - x
        ks = self.base_kernel(np.linalg.norm(dx, axis=1) / self.bw) / self.bw
        
        mu0 = ks.mean()
        mu1 = ks.dot(dx) / N
        mu2 = (ks[:, None] * dx).T.dot(dx) / N
        sig2 = mu0 * mu2 - mu1**2

        weights = ks[:, None] * (mu2 - mu1*dx) / sig2
        return self._normalize_weights(weights[:,0], sum_to_one=True, clip=True, clip_allow_neg=True)

        # REMOVE THIS IMPLEMENTATION UNTIL WE HAVE SMT FOR p>1
        # N = self.x_train.shape[0]
        # mu1 = ks.dot(dx) / N
        # mu2 = (ks[:, None] * dx).T.dot(dx) / N
        # mu2_chol = cho_factor(mu2)
        # weights = np.sum(ks[:, None] * (1 - mu1) * cho_solve(mu2_chol, dx.T).T, axis=1)
        # return self._normalize_weights(weights[:,0], sum_to_one=True, clip=True, clip_allow_neg=True)
