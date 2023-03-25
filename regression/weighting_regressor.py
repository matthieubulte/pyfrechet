from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
from metric_spaces import MetricData

TWeightingRegressor = TypeVar("TWeightingRegressor", bound="WeightingRegressor")

class WeightingRegressor(metaclass=ABCMeta):

    @abstractmethod
    def _weights_for(self, x) -> np.ndarray:
        pass

    def _predict_one(self, x):        
        return self.y_train.frechet_mean(self.weights_for(x))
    
    @abstractmethod
    def fit(self, X, y: MetricData) -> TWeightingRegressor:
        self.y_train = y
        return self
    
    def weights_for(self, x) -> np.ndarray:
        weights = self._weights_for(x)
        weights = np.clip(weights, a_min=np.finfo(weights.dtype).eps, a_max=None)
        return weights / weights.sum()

    def predict(self, x):
        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._predict_one(x)
        else:
            y0 = self._predict_one(x[0,:])
            y_pred = np.zeros((x.shape[0], y0.shape[0]))
            y_pred[0,:] = y0
            for i in range(1, x.shape[0]):
                y_pred[i,:] = self._predict_one(x[i,:])
            return MetricData(self.y_train.M, y_pred)
        
    @abstractmethod
    def clone(self) -> TWeightingRegressor:
        pass