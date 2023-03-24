# https://scholar.google.de/scholar?start=30&q=adaboost+theoretical+guarantees&hl=en&as_sdt=0,5&as_vis=1

import numpy as np
from mrf import MetricTree
from metric_spaces.utils import mat_sel_idx

class AdaBoost:
    def __init__(self, n_estimators, learning_rate, M):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.M = M

    def _boost(self, X, y, D, sample_weight, is_last):
        bootstrap_idx = np.random.choice(
            np.arange(X.shape[0]),
            size=X.shape[0],
            replace=True,
            p=sample_weight
        )
        X_ = X[bootstrap_idx, :]
        D_ = mat_sel_idx(D, bootstrap_idx)

        # this might cause a bug at prediction because of how the selection work!
        estimator = MetricTree().fit(X_, D_, np.ones(X.shape[0]))

        def __pred(i):
            w = estimator.select_for(X[i, :])
            return self.M.frechet_mean(y, w=w)

        # todo: vectorize
        error_vect = [ self.M.d(self.M.index(y, i), __pred(i))**2 for i in range(X.shape[0]) ]
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

    def fit(self, X, y, D):
        sample_weight = np.ones(X.shape[0]) / X.shape[0]
        
        self.y_train = y
        self.estimators = []
        self.estimator_weights = np.zeros(self.n_estimators, dtype=np.float64)

        epsilon = np.finfo(sample_weight.dtype).eps
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            
            # Boosting step
            estimator, sample_weight, estimator_weight, estimator_error = self._boost(X, y, D, sample_weight, iboost+1==self.n_estimators)

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

    def get_weights(self, x):
        y_weights = np.zeros(self.y_train.shape[0])
        for (i, tree) in enumerate(self.estimators):
            tree_sel = tree.select_for(x)+0
            y_weights += self.estimator_weights[i] * tree_sel / np.sum(tree_sel)
        return y_weights

    def predict(self, x):
        return self.M.frechet_mean(self.y_train, w=self.get_weights(x))

