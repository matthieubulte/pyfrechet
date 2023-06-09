from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional, Tuple, List
import sklearn

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


class BaggedRegressor(WeightingRegressor):
    def __init__(self, 
                 estimator: Optional[WeightingRegressor] = None,
                 n_estimators: int = 100,
                 bootstrap_fraction: float = 0.75,
                 n_jobs=-2):
        super().__init__()
        self.estimator = estimator
        self.precompute_distances = estimator.precompute_distances if estimator else False
        self.n_estimators = n_estimators
        self.estimators:List[Tuple[np.ndarray, WeightingRegressor]] = []
        self.bootstrap_fraction = bootstrap_fraction
        self.n_jobs = n_jobs

    def _make_mask(self, N):
        s = int(self.bootstrap_fraction * N)
        return np.random.choice(N, s, replace=False)

    def _fit_est(self, X, y):
        mask = self._make_mask(X.shape[0])
        return (mask, sklearn.clone(self._estimator).fit(X[mask, :], y[mask]))

    def _fit_par(self, X, y: MetricData):
        super().fit(X, y)
        def calc(): return self._fit_est(X, y)
        self.estimators = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(calc)() for _ in range(self.n_estimators)) or []
        return self

    def _fit_seq(self, X, y: MetricData):
        super().fit(X, y)
        self.estimators = [ self._fit_est(X, y) for _ in tqdm(range(self.n_estimators))]
        return self

    def fit(self, X, y: MetricData):
        super().fit(X, y)
        
        assert self.estimator
        self._estimator = self.estimator
        
        return self._fit_seq(X, y) if self.n_jobs == 1 or not self.n_jobs else self._fit_par(X, y)

    def weights_for(self, x) -> np.ndarray:
        assert len(self.estimators) > 0
        weights = np.zeros(self.y_train_.shape[0])
        for (mask, estimator) in self.estimators:
            est_weights = estimator.weights_for(x)
            weights[mask] += est_weights
        return self._normalize_weights(weights / self.n_estimators, clip=True)
