from abc import ABCMeta, abstractmethod
from typing import TypeVar

from metric_spaces import MetricData

TWeightingRegressor = TypeVar("TWeightingRegressor", bound="WeightingRegressor")

class WeightingRegressor(metaclass=ABCMeta):
    def __init__(self):
        self.y_train = None

    @abstractmethod
    def fit(self, X, y: MetricData) -> TWeightingRegressor:
        self.y_train = y
        return self

    @abstractmethod
    def weights_for(self, x):
        pass

    def predict(self, x):
        return self.y_train.frechet_mean(self.weights_for(x))

    @abstractmethod
    def clone(self) -> TWeightingRegressor:
        pass