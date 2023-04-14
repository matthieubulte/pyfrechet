from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from metric_spaces import MetricData
from metrics import mean_squared_error
from sklearn.metrics._regression import _assemble_r2_explained_variance

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
        X, y = check_X_y(X, y)
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
        
    def score(self, X, y: MetricData, sample_weight=None, multioutput="uniform_average", force_finite=True,):
        y_pred = self.predict(X)
        y_bar = y.frechet_mean(sample_weight)
        mse = mean_squared_error(y.M)

        numerator = mse(y_true, y_pred, sample_weight=sample_weight)
        denominator = mse(y_true, y_bar, sample_weight=sample_weight)

        return _assemble_r2_explained_variance(
            numerator=numerator,
            denominator=denominator,
            n_outputs=y_true.shape[1],
            multioutput=multioutput,
            force_finite=force_finite,
        )

        # base_err = 

