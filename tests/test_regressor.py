if __name__ == "__main__":
    from context import *

import pytest
import numpy as np
from sklearn.base import clone

from pyfrechet.metric_spaces import *
from pyfrechet.regression.knn import KNearestNeighbours
from pyfrechet.regression.kernels import NadarayaWatson
from pyfrechet.regression.frechet_regression import GlobalFrechet, LocalFrechet
from pyfrechet.metrics import mse

def gen_sphere(N, m_type='contant', eps=0.1):
    M = Sphere(1)
    
    m = lambda _: 1
    if m_type == 'constant':
        m = lambda _: 1
    elif m_type == 'linear':
        m = lambda x: 0.5 + 2*x[:,0]
    elif m_type == 'nonlinear':
        m = lambda x: 5 + 5*x[:,0] - 10*x[:,0]**2

    x = np.random.rand(N*1).reshape((N,1))
    theta = m(x) + eps*np.random.randn(N)
    y = np.c_[np.cos(theta), np.sin(theta)]
    return x, MetricData(M, y)


REGRESSORS = [KNearestNeighbours(n_neighbors=10), NadarayaWatson(bw=0.1), LocalFrechet(bw=0.1), GlobalFrechet()]


@pytest.mark.parametrize("regressor", REGRESSORS)
def test_can_fit_and_predict(regressor):
    X, y = gen_sphere(20, m_type='contant')
    fitted = clone(regressor).fit(X, y)
    fitted.predict(np.array([0.5]).reshape(1, -1))


@pytest.mark.parametrize("regressor", REGRESSORS)
def test_can_fit_constant(regressor):
    X_train, y_train = gen_sphere(1000, m_type='contant')
    X_test, y_test = gen_sphere(100, m_type='contant')

    fitted = clone(regressor).fit(X_train, y_train)
    assert mse(fitted.predict(X_test), y_test) < 0.1


@pytest.mark.parametrize("regressor", REGRESSORS)
def test_can_fit_linear(regressor):
    X_train, y_train = gen_sphere(1000, m_type='linear')
    X_test, y_test = gen_sphere(100, m_type='linear')

    fitted = clone(regressor).fit(X_train, y_train)
    assert mse(fitted.predict(X_test), y_test) < 0.1


@pytest.mark.parametrize("regressor", [KNearestNeighbours(n_neighbors=10), NadarayaWatson(bw=0.05), LocalFrechet(bw=0.05)])
def test_can_fit_nonlinear(regressor):
    X_train, y_train = gen_sphere(10_000, m_type='nonlinear')
    X_test, y_test = gen_sphere(100, m_type='nonlinear')

    fitted = clone(regressor).fit(X_train, y_train)
    assert mse(fitted.predict(X_test), y_test) < 0.3


if __name__ == "__main__":
    pytest.main(["tests/test_regressor.py"])
