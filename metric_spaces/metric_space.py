from abc import ABCMeta, abstractmethod
import numpy as np
from .utils import coalesce_weights

class MetricSpace(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def d(self, x, y) -> float:
        pass

    @abstractmethod
    def _frechet_mean(self, y, w=None):
        pass

    def frechet_mean(self, y, w=None):
        w = coalesce_weights(w, y)
        return self._frechet_mean(y, w)
    
    def frechet_var(self, y, w=None):
        y_bar = self.frechet_mean(y, w=w)
        return np.sum([ self.d(y_bar, self.index(y, i))**2 for i in range(y.shape[0]) ])

    def index(self, y, i):
        return y[i, :]