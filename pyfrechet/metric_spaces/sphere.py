import autograd.numpy as anp
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from .metric_space import MetricSpace

class Sphere(MetricSpace):
    def __init__(self, dim):
        self.dim = dim

    def _d(self, x, y):
        return anp.maximum(-1, anp.minimum(1, anp.arccos(anp.dot(x,y.T)).sum(axis=0)))
    
    def _frechet_mean(self, y, w):
        manifold = pymanopt.manifolds.Sphere(self.dim)

        @pymanopt.function.autograd(manifold)
        def cost(om): return anp.dot(w, self._d(om.reshape((1,self.dim)), y))

        problem = pymanopt.Problem(manifold, cost)
        optimizer = pymanopt.optimizers.TrustRegions(verbosity=0)

        # initial y 
        y_init = w.dot(y)
        # add a small perturbation to y_init to make sure we don't land in the sample
        y_init += 1e-4 * np.random.randn(self.dim)
        y_init /= np.linalg.norm(y_init)

        result = optimizer.run(problem, initial_point=y_init)
        return result.point

    def __str__(self):
        return f'Sphere(dim={self.dim})'
 
def r2_to_angle(x):
    angles = np.arctan2(x[:,1], x[:,0])
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    return angles