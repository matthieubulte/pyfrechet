import numpy as np
import numpy as np
from .weighting_regressor import WeightingRegressor
from pyfrechet.metric_spaces import MetricData


def epanechnikov(u):
    return 3.0 * (1.0 - np.minimum(1, u**2))/4.0


def gaussian(u):
    return np.exp(-u**2/2) /np.sqrt(2*np.pi)


class NadarayaWatson(WeightingRegressor):
    def __init__(self, base_kernel=gaussian, bw=1.0):
        super().__init__()
        self.base_kernel = base_kernel
        self.bw = bw

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.x_train_ = X
        return self

    def weights_for(self, x):
        weights = self.base_kernel(np.linalg.norm(x - self.x_train_, axis=1) / self.bw)
        return self._normalize_weights(weights, sum_to_one=True, clip=True)
