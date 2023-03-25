from tqdm import tqdm

from metric_spaces import MetricData
from metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


class BaggedRegressor(WeightingRegressor):
    def __init__(self, estimator: WeightingRegressor, n_estimators: int, bootstrap_fraction: float, n_jobs=-2):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.bootstrap_fraction = bootstrap_fraction
        self.n_jobs = n_jobs

    def _make_mask(self, N):
        s = int(self.bootstrap_fraction * N)
        mask = np.repeat(False, N)
        mask[np.random.choice(N, s, replace=False)] = True
        return mask

    def _fit_par(self, X, y: MetricData):
        super().fit(X, y)
        def calc(mask): return self.estimator.clone().fit(X, y, basemask=mask)
        self.estimators = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(calc)(self._make_mask(X.shape[0])) for _ in range(self.n_estimators)) or []
        return self

    def _fit_seq(self, X, y: MetricData):
        super().fit(X, y)
        it = tqdm(range(self.n_estimators))
        self.estimators = [self.estimator.clone().fit(X, y, basemask=self._make_mask(X.shape[0])) for _ in it]
        return self

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        return self._fit_seq(X, y) if self.n_jobs == 1 or not self.n_jobs else self._fit_par(X, y)

    def _weights_for(self, x):
        assert len(self.estimators) > 0
        weights = self.estimators[0].weights_for(x)
        for estimator in self.estimators[1:]:
            est_weights = estimator.weights_for(x)
            weights += est_weights
        return weights

    def clone(self):
        return type(self)(self.estimator, self.n_estimators, self.bootstrap_fraction, self.n_jobs)
