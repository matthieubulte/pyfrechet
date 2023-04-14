import numpy as np
from geomstats.learning.frechet_mean import FrechetMean
from .metric_space import MetricSpace

"""
MetricSpace implementation for Riemannian manifolds from Geomstats
"""
class RiemannianManifold(MetricSpace):
    def __init__(self, manifold):
        self.manifold = manifold
    
    def _d(self, x, y):
        return self.manifold.metric.dist(x, y)
    
    def _frechet_mean(self, y, w):
        mean = FrechetMean(metric=self.manifold.metric)
        mean.fit(y, weights=w)
        return mean.estimate_

    def add_normal_noise(self, x, sig):
        v = sig * np.random.normal(size=(self.manifold.dim+1,))
        pv = v - self.manifold.metric.inner_product(x, x, v) * x
        return self.manifold.metric.exp(pv, x)

    def __str__(self):
        return f'Manifold(manifold={self.manifold})'
    