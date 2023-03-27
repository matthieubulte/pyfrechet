import numpy as np
from typing import Union, Any, TypeVar
import numbers
from .utils import D_mat_par, D_mat, mat_sel_idx, mat_sel, coalesce_weights

T = TypeVar("T", bound="MetricData")

class MetricData:
    def __init__(self, M, data, distances=None):
        self.M = M
        self.data = data
        self.distances = distances
        self.shape = (data.shape[0],)

    def compute_distances(self, n_jobs=-2):
        if self.distances is None:
            if n_jobs is None or n_jobs == 1:
                self.distances = D_mat(self.M, self.data)
            else:
                self.distances = D_mat_par(self.M, self.data, n_jobs)

    def frechet_mean(self, weights=None):
        return self.M.frechet_mean(self.data, weights)

    def frechet_var(self, weights=None):
        return self.M.frechet_var(self.data, weights)
    
    def frechet_medoid(self, weights=None, n_jobs=-2):
        self.compute_distances(n_jobs=n_jobs)
        weights = coalesce_weights(weights, self)
        idx = np.argmin(self.distances.dot(weights))
        return self[idx]

    def frechet_medoid_var(self, weights=None, n_jobs=-2):
        self.compute_distances(n_jobs=n_jobs)
        weights = coalesce_weights(weights, self)
        return np.min(self.distances.dot(weights))

    def __getitem__(self, key) -> Union[Any, T]:
        subset = self.M.index(self.data, key)
        if isinstance(key, numbers.Integral):
            return subset
        elif self.distances is None:
            return MetricData(self.M, subset)
        else:
            key = key if type(key) is np.ndarray else np.array(key)
            subdist = mat_sel(self.distances, key) if key.dtype == 'bool' else mat_sel_idx(self.distances, key)
            return MetricData(self.M, subset, subdist)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __str__(self):
        return f'MetricData(M={self.M}, len={len(self)}, has_distance={not self.distances is None})'