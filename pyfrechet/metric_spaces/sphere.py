import numpy as np
from .riemannian_manifold import RiemannianManifold
from geomstats.geometry.hypersphere import Hypersphere

class Sphere(RiemannianManifold):
    def __init__(self, dim):
        super().__init__(Hypersphere(dim=dim))

    def __str__(self):
        return f'Sphere(dim={self.manifold.dim})'
 
def r2_to_angle(x):
    return Hypersphere(dim=1).extrinsic_to_angle(x)

def r3_to_angles(x):
    return Hypersphere(dim=2).extrinsic_to_spherical(x)