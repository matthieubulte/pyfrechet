from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional
import sklearn
from scipy.sparse import coo_array

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


class BaggedRegressor(WeightingRegressor):
    def __init__(self, 
                 estimator: Optional[WeightingRegressor] = None,
                 n_estimators: int = 100,
                 bootstrap_fraction: float = 0.75,
                 n_jobs=-2):
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

    def _fit_est(self, X, y):
        mask = self._make_mask(X.shape[0])
        return (coo_array(mask), sklearn.clone(self.estimator).fit(X[mask, :], y[mask]))

    def _fit_par(self, X, y: MetricData):
        super().fit(X, y)
        def calc(): return self._fit_est(X, y)
        self.estimators = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(calc)() for _ in range(self.n_estimators)) or []
        return self

    def _fit_seq(self, X, y: MetricData):
        super().fit(X, y)
        it = tqdm(range(self.n_estimators))
        self.estimators = [ self._fit_est(X, y) for _ in it]
        return self

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        return self._fit_seq(X, y) if self.n_jobs == 1 or not self.n_jobs else self._fit_par(X, y)

    def weights_for(self, x):
        assert len(self.estimators) > 0
        weights = np.zeros(self.estimators[0][0].shape[1])
        for (sp_mask, estimator) in self.estimators:
            est_weights = estimator.weights_for(x)
            mask = sp_mask.toarray()[0,:]
            weights[mask] += est_weights
        return self._normalize_weights(weights / self.n_estimators, clip=True)
