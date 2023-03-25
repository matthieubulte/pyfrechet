import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import numpy as np
from metric_spaces import MetricData
from metric_spaces.sphere import Sphere, r2_to_angle
from regression.frechet_regression import LocalFrechet, GlobalFrechet
from regression.kernels import epanechnikov, gaussian

m = lambda x: 5 + 5*x[:,0]**2 + np.sin(20 * x[:,0]) - 10*x[:,0]**3 

def gen_data(N, eps=0.1):
    x = np.random.rand(N*1).reshape((N,1))

    theta = m(x) + eps*np.random.randn(N)
    y = np.c_[np.cos(theta), np.sin(theta)]
    return x, MetricData(Sphere(1), y)

N = 200; eps = 0.5
x_train, y_train = gen_data(N, eps)
x_test, y_test = gen_data(100, eps)

global_frechet = GlobalFrechet().fit(x_train, y_train)
local_frechet = LocalFrechet(gaussian, 0.02).fit(x_train, y_train)

preds = [global_frechet.predict(x_test), local_frechet.predict(x_test)]

errs = np.zeros((2, x_test.shape[0]))
errs[0,:] = Sphere(1).d(y_test, preds[0])
errs[1,:] = Sphere(1).d(y_test, preds[1])