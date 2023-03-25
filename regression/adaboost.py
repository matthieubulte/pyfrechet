# https://scholar.google.de/scholar?start=30&q=adaboost+theoretical+guarantees&hl=en&as_sdt=0,5&as_vis=1

import numpy as np

from metric_spaces import MetricData
from .weighting_regressor import WeightingRegressor, TWeightingRegressor


class AdaBoost(WeightingRegressor):
    def __init__(self, estimator:WeightingRegressor, n_estimators, learning_rate):
        self.learning_rate = learning_rate
        self.estimator = estimator
        self.estimators = []
        self.estimator_weights = None
        self.n_estimators = n_estimators

    def _boost(self, X, y: MetricData, sample_weight, is_last):
        bootstrap_idx = np.random.choice(
            np.arange(X.shape[0]),
            size=X.shape[0],
            replace=True,
            p=sample_weight
        )
        X_ = X[bootstrap_idx, :]
        y_ = y[bootstrap_idx]

        # this might cause a bug at prediction because of how the selection work!
        estimator = self.estimator.clone().fit(X_, y_)

        def __pred(i):
            # be careful here because the weights don't match
            w = estimator.weights_for(X[i, :])
            return y.frechet_mean(w)

        # todo: vectorize
        error_vect = [ y.M.d(y[i], __pred(i))**2 for i in range(X.shape[0]) ]
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        
        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return estimator, sample_weight, 1.0, 0.0

        elif estimator_error >= 0.5:
            return None, None, None, None

        beta = estimator_error / (1.0 - estimator_error)
        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1.0 / beta)

        if not is_last:
            sample_weight[sample_mask] *= np.power(
                beta, (1.0 - masked_error_vector) * self.learning_rate
            )

        return estimator, sample_weight, estimator_weight, estimator_error

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        sample_weight = np.ones(X.shape[0]) / X.shape[0]
        self.estimators = []
        self.estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)

        epsilon = np.finfo(sample_weight.dtype).eps
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            
            # Boosting step
            estimator, sample_weight, estimator_weight, estimator_error = self._boost(X, y, sample_weight, iboost+1==self.n_estimators)

            # Early termination
            if sample_weight is None:
                break

            self.estimators.append(estimator)
            self.estimator_weights[iboost] = estimator_weight

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                print(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    def weights_for(self, x):
        y_weights = np.zeros(len(self.y_train))
        for (i, estimator) in enumerate(self.estimators):
            est_weights = estimator.weights_for(x)+0
            y_weights += self.estimator_weights[i] * est_weights / np.sum(est_weights)
        return y_weights

    def clone(self) -> TWeightingRegressor:
        return AdaBoost(self.estimator, self.n_estimators, self.learning_rate)