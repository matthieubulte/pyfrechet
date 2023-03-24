import autograd.numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers


def d(x, y):
    return np.sum(np.square(np.arccos(np.dot(x,y.T))), axis=0)

def frechet_mean(y, w):
    manifold = pymanopt.manifolds.Sphere(2)

    @pymanopt.function.autograd(manifold)
    def cost(om):
        return np.dot(w, d(om.reshape((1,2)), y))

    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result = optimizer.run(problem)
    return result.point

 
def r2_to_angle(x):
    angles = np.arctan2(x[:,1], x[:,0])
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    return angles