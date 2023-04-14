import numpy as np
from sklearn.neighbors import NearestNeighbors
from .weighting_regressor import WeightingRegressor
from pyfrechet.metric_spaces import MetricData

class KNearestNeighbours(WeightingRegressor):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.nn.fit(X)
        return self

    def weights_for(self, x):
        mask = np.zeros(len(self.y_train_))
        neighbors_idx = self.nn.kneighbors(np.array(x).reshape(1, -1), self.n_neighbors, False)[0]
        mask[neighbors_idx] = 1.0
        return self._normalize_weights(mask, sum_to_one=True)