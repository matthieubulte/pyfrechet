from sklearn.metrics import make_scorer
import numpy as np
from metric_spaces.utils import coalesce_weights
from metric_spaces import MetricSpace, MetricData

def mean_squared_error(M: MetricSpace):
    def _mse(y_true: MetricData, y_pred: MetricData, sample_weights=None):
        return np.dot(coalesce_weights(sample_weights, y_true), M.d(y_true, y_pred)**2)
    return make_scorer(_mse, greater_is_better=False)