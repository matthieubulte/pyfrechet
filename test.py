import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import autograd.numpy as np
from metric_spaces import MetricData
from metric_spaces.sphere import Sphere
from regression.forests import RandomForest 
from regression.trees import MedoidTree


def gen_data(N, eps=0.1, d=3):
    assert d >= 3
    x = np.random.rand(N*d).reshape((N,d))
    m = lambda x: 5 + 5*x[:,0]**2 + np.sin(20 * x[:,1]) - 10*x[:,2]**3 

    theta = m(x) + eps*np.random.randn(N)
    y = np.c_[np.cos(theta), np.sin(theta)]
    return x, MetricData(Sphere(1), y)


# It seems that t
# - the NW estimator is better than RF for d=1 (in the simple model)
# - for d=3 (=true d) the random forest is slightly better
# - for d large (true d=3), the random forest is much better

N = 50; eps = 0.5; d=3
x_train, y_train = gen_data(N, eps, d)
x_test, y_test = gen_data(100, eps, d)

y_train.compute_distances()

# Observations:
#     - Experiment with s < 0.5*N had not great results because weights are not local enough
#     - Works well with small data as long as s is large enough
#     - s has the most influence on speed
# forest = rf(1000, x_train, D, 250)
forest = RandomForest(MedoidTree(), 100, 0.75).fit(x_train, y_train)

print(forest.estimators[0].weights_for(x_test[0,:]).sum())
print(forest.weights_for(x_test[0,:]).sum())

# preds = np.zeros((2, x_test.shape[0], 2))
# for i in range(x_test.shape[0]):
#     preds[0,i,:] = circle.fm(y_train, rf_weights(forest, x_test[i,:]))
#     preds[1,i,:] = circle.fm(y_train, nw_weights(x_test[i,:], x_train, gaussian, 0.02))