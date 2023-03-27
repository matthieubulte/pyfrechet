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
        x_d_y = np.dot(x,y.T)
        if x_d_y > 1.0 - np.finfo(x.dtype).eps:
            return 0.0
        elif x_d_y < -1.0 + np.finfo(x.dtype).eps:
            return np.pi
        
        return np.sqrt(np.sum(np.square(np.arccos(x_d_y)), axis=0))
    
    def _frechet_mean(self, y, w):
        manifold = pymanopt.manifolds.Sphere(self.dim)

        def _d(x, y): return anp.sum(anp.square(anp.arccos(anp.dot(x,y.T))), axis=0)
        
        @pymanopt.function.autograd(manifold)
        def cost(om): return anp.dot(w, _d(om.reshape((1,self.dim)), y))

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