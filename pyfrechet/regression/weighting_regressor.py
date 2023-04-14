from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pyfrechet.metric_spaces import MetricData
from pyfrechet.metrics import r2_score

T = TypeVar("T", bound="WeightingRegressor")

class WeightingRegressor(RegressorMixin, BaseEstimator, metaclass=ABCMeta):

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
        return self.y_train_.frechet_mean(self.weights_for(x))
    
    @abstractmethod
    def fit(self:T, X, y: MetricData) -> T:
        X, _ = check_X_y(X, y.data, multi_output=True)
        self.y_train_ = y
        return self
    
    @abstractmethod
    def weights_for(self, x) -> np.ndarray:
        pass

    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x)

        if len(x.shape) == 1 or x.shape[0] == 1:
            return self._predict_one(x)
        else:
            y0 = self._predict_one(x[0,:])
            y_pred = np.zeros((x.shape[0], y0.shape[0]))
            y_pred[0,:] = y0
            for i in range(1, x.shape[0]):
                y_pred[i,:] = self._predict_one(x[i,:])
            return MetricData(self.y_train_.M, y_pred)
        
    def score(self, X, y: MetricData, sample_weight=None, force_finite=True,):
        return r2_score(y, self.predict(X), sample_weight=sample_weight, force_finite=force_finite)