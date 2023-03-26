from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
from src.metric_spaces import MetricData

T = TypeVar("T", bound="WeightingRegressor")

class WeightingRegressor(metaclass=ABCMeta):

    def _normalize_weights(self, weights, sum_to_one=False, clip=False, clip_allow_neg=False):
        if sum_to_one:
            weights /= weights.sum()
        
        if clip:
            eps = np.finfo(weights.dtype).eps
            if clip_allow_neg:
                clipped = np.clip(np.abs(weights), a_min=eps, a_max=None)
                weights[clipped == eps] = 0.0
            else:
                weights = np.clip(weights, a_min=eps, a_max=None)
                weights[weights == eps] = 0.0
        
        if sum_to_one:
            weights /= weights.sum()
        
        return weights

    def _predict_one(self, x):        
        return self.y_train.frechet_mean(self.weights_for(x))
    
    @abstractmethod
    def fit(self:T, X, y: MetricData) -> T:
        self.y_train = y
        return self
    
    @abstractmethod
    def weights_for(self, x) -> np.ndarray:
        pass

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
    def clone(self:T) -> T:
        pass