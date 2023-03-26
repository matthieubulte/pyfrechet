import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import matplotlib.pyplot as plt
from ipywidgets import *

import numpy as np
from metric_spaces import MetricData
from metric_spaces.sphere import Sphere, r2_to_angle
from regression.frechet_regression import LocalFrechet, GlobalFrechet
from regression.kernels import epanechnikov, gaussian
from metric_spaces.euclidian import Euclidian

M = Euclidian(1)
m = lambda x: 5 + 5*x[:,0]**2 + np.sin(20 * x[:,0]) - 10*x[:,0]**3 

def gen_data(N, eps=0.1):
    x = np.random.rand(N*1).reshape((N,1))

    theta = m(x) + eps*np.random.randn(N)
    return x, MetricData(M, theta.reshape((N,1)))
#     y = np.c_[np.cos(theta), np.sin(theta)]
#     return x, MetricData(M, y)

N = 500; eps = 0.5
x_train, y_train = gen_data(N, eps)
x_test, y_test = gen_data(100, eps)

for bw in [0.0005]:#, 0.001, 0.005, 0.01, 0.02]:
# for bw in [0.005, 0.01, 0.02]:
    local_frechet = LocalFrechet(gaussian, bw).fit(x_train, y_train)
    preds = local_frechet.predict(x_test)
    errs = M.d(y_test, preds)**2
    print(bw, errs[~np.isnan(errs)].mean(), np.isnan(errs).sum())
    
# fr_angles = r2_to_angle(preds.data)
# theta_test = r2_to_angle(y_test.data)

# plt.scatter(x_test, theta_test, c='red',s=1,label="Y_i")
# plt.scatter(x_test, fr_angles, c='green',s=1,label="\\hat Y_i (local)")
# plt.plot(np.linspace(0,1,100), m(np.linspace(0,1,100).reshape((100,1))), c='black', linestyle='--', label='m(x)', alpha=0.1)
# plt.legend()