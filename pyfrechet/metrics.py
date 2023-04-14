import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics._regression import _assemble_r2_explained_variance
from pyfrechet.metric_spaces.utils import coalesce_weights
from pyfrechet.metric_spaces import MetricSpace, MetricData

def mse(y_true: MetricData, y_pred: MetricData, sample_weight=None):
    return np.dot(coalesce_weights(sample_weight, y_true), y_true.M.d(y_true, y_pred)**2)

def r2_score(y_true, y_pred, sample_weight=None, force_finite=True):
    y_bar = y_true.frechet_mean(sample_weight)
    numerator = mse(y_true, y_pred, sample_weight=sample_weight)
    denominator = mse(y_true, y_bar, sample_weight=sample_weight)

    return _assemble_r2_explained_variance(
        numerator=numerator,
        denominator=denominator,
        n_outputs=1,
        multioutput=None,
        force_finite=force_finite,
    )

mean_squared_error = make_scorer(mse, greater_is_better=False)