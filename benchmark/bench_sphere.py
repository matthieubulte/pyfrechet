import sys, os;

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname('/Users/matthieubulte/Documents/university/phd/merf'))

import numpy as np
from scipy import stats
from benchmark import bench
from pyfrechet.metric_spaces import *

OUT_FILE = 'result_sphere_cross_est_new_dgp_2.json'

def gen_data(N, p, alpha, beta, eps=0.1):
    M = Sphere(2)

    def m(x):
        eta = 2 * (x - 0.5).dot(beta) / np.sqrt(p) + alpha
        z = stats.logistic.cdf(eta)
        dz = np.sqrt(1 - z**2)
        pz = np.pi * z
        y= np.c_[dz * np.cos(pz), dz * np.sin(pz), z]
        return y

    def add_noise(x, sig):
        v = sig * np.random.normal(size=(M.manifold.dim+1,))
        pvx = v - np.dot(x, v) * x
        return M.manifold.metric.exp(pvx, x)

    x = np.random.rand(N*p).reshape((N,p))
    mx = m(x)
    y = np.array([ add_noise(mx[i,:], eps) for i in range(N)])
    return x, MetricData(M, y), MetricData(M, mx)

bench(
    gen_data,
    OUT_FILE,
    ps=[50, 200, 400, 800],
    Ns=[5],
    replicas=25
)