import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import circle
import distance_utils
import mrf
import nw

def epanechnikov(u):
    return 3.0 * (1.0 - np.minimum(1, u**2))/4.0

def nw_weights(x, X, K, bw):
    w = K(np.linalg.norm(x - X, axis=1) / bw)
    return w / w.sum()

def gen_data(N, eps=0.1):
    x = np.random.rand(N)
    m = lambda x: 5 + 5*x**2 + np.sin(20 * x) - 10*x**3 

    theta = m(x) + eps*np.random.randn(N)
    y = np.c_[np.cos(theta), np.sin(theta)]
    x = x.reshape((N,1))
    return x, y

N = 50; eps = 0.5
x_train, y_train = gen_data(N, eps)
x_test, y_test = gen_data(2*N, eps)

D = distance_utils.D_mat(circle.d, y_train)
forest = mrf.rf(1000, x_train, D, 30)

bw_grid = np.linspace(0.01,0.1,10)

errs = np.zeros((bw_grid.shape[0] + 1, x_test.shape[0]))
for i in tqdm(range(x_test.shape[0])):
    errs[0,i] = circle.d(
        y_test[i,:],
        circle.fm(y_train, mrf.rf_weights(forest, x_test[i,:]))
    )
    for j in range(bw_grid.shape[0]):
        errs[j+1,i] = circle.d(
            y_test[i,:], 
            circle.fm(y_train, nw_weights(x_test[i,:], x_train, epanechnikov, bw_grid[j])))


errs.mean(axis=1)


