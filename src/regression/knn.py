import numpy as np
from sklearn.neighbors import NearestNeighbors
from .weighting_regressor import WeightingRegressor
from src.metric_spaces import MetricData

class KNearestNeighbours(WeightingRegressor):
    def __init__(self, k):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k)

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        self.nn.fit(X)
        return self

    def weights_for(self, x):
        mask = np.zeros(len(self.y_train))
        neighbors_idx = self.nn.kneighbors([x], self.k, False)[0]
        mask[neighbors_idx] = 1.0
        return self._normalize_weights(mask, sum_to_one=True)
    
    def clone(self):
        return KNearestNeighbours(self.k)