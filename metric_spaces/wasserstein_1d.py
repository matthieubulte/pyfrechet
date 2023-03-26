import numpy as np
from .metric_space import MetricSpace
from sklearn.isotonic import isotonic_regression

class Wasserstein1D(MetricSpace):
    GRID = np.linspace(0, 1, 102)[1:-1]

    def __init__(self):
        pass

    def _d(self, x, y):
        return np.sqrt(np.trapz((x - y)**2, Wasserstein1D.GRID))
    
    def _frechet_mean(self, y, w):
        # Computed the (weighted) averaged quantiles, then project to make sure that q is increasing
        return isotonic_regression(np.dot(w, y))

    def __str__(self):
        return 'Wasserstein'

def noise(J=5, grid=Wasserstein1D.GRID):
    def _T(K): return grid if K == 0 else grid - np.sin(grid * np.pi * K) / (np.pi * np.abs(K))
    def _K(): return 2 * (np.random.binomial(1,0.5) - 1)*np.random.poisson(3)
    U = np.sort(np.random.uniform(size=J-1))
    T = np.array([ _T(_K()) for _ in range(J) ])
    return U[0] * T[0,:] + np.dot(U[1:] - U[:-1], T[1:-1, :]) + (1 - U[-1]) * T[-1,:]
