import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.hypersphere import Hypersphere

class Sphere(RiemannianManifold):
    def __init__(self, dim):
        super().__init__(Hypersphere(dim=dim))

    def __str__(self):
        return f'Sphere(dim={self.manifold.dim})'
 
def r2_to_angle(x):
    angles = np.arctan2(x[:,1], x[:,0])
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    return angles