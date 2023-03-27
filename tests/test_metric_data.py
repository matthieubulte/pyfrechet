if __name__ == "__main__":
    from context import *

import pytest
import skfda
import numpy as np
from scipy.stats import beta, norm
from pyfrechet.metric_spaces import *

def gen_euclidean(n): return Euclidean(5), np.random.rand(n*5).reshape((n, 5))

def gen_matrices(n):
    dim = 5
    y = np.random.rand(n*dim*dim).reshape((n,dim,dim))
    y = np.array([ y[i,:,:].T.dot(y[i,:,:]) for i in range(n) ])
    return CorrFroebenius(5), y

def gen_sphere(n): 
    dim = 4
    y = np.random.rand(n*dim).reshape((n, dim))
    return Sphere(dim), y / np.linalg.norm(y, axis=1)[:,None]

def gen_fr_phase(n):
    # normal pdf with random mean/var
    grid = np.linspace(0, 1, 100)
    rand_mu = lambda: np.random.rand()
    rand_sig = lambda: np.random.randn()**2
    y_data = np.array([  norm.pdf(grid, rand_mu(), rand_sig()) for _ in range(n) ])
    return FisherRaoPhase(grid), skfda.FDataGrid(y_data, grid)

def gen_wasserstein(n):
    # beta quantiles with random parameters
    rand_a = lambda: np.random.exponential()
    rand_b = lambda: np.random.exponential()
    y = np.array([ beta.ppf(Wasserstein1D.GRID, rand_a(), rand_b()) for _ in range(n) ])
    return Wasserstein1D(), y

TOL = 1e-6
GENERATORS = [
    gen_sphere,
    gen_euclidean,
    gen_matrices,
    gen_wasserstein,
    gen_fr_phase
]

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_init(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)

    assert data.shape == (10,)
    assert len(data) == 10
    assert data.M is M
    assert data.distances is None

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_getitem__single(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)

    assert M.d(data[1], M.index(y, 1)) < TOL    

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_getitem__slice(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    subdata = data[:2]

    assert subdata.shape == (2,)
    assert len(subdata) == 2
    assert M.d(subdata[0], data[0]) < TOL
    assert M.d(subdata[1], data[1]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_getitem__multiindex(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    subdata = data[[0, 2, 4]]

    assert subdata.shape == (3,)
    assert len(subdata) == 3
    assert M.d(subdata[0], data[0]) < TOL
    assert M.d(subdata[1], data[2]) < TOL
    assert M.d(subdata[2], data[4]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_getitem__mask(gen_data):
    M, y = gen_data(4)
    data = MetricData(M, y)
    subdata = data[[True, False, True, False]]

    assert subdata.shape == (2,)
    assert len(subdata) == 2
    assert M.d(subdata[0], data[0]) < TOL
    assert M.d(subdata[1], data[2]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_getitem__distances(gen_data):
    M, y = gen_data(4)
    data = MetricData(M, y)
    data.compute_distances(n_jobs=None)

    subdata = data[[True, False, True, False]]

    assert subdata.shape == (2,)
    assert len(subdata) == 2
    assert not subdata.distances is None
    assert subdata.distances[0,1] == data.distances[0,2]

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_compute_distances(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    data.compute_distances(n_jobs=None)

    assert not data.distances is None
    assert data.distances.shape == (10,10)
    assert np.all(data.distances.T == data.distances)

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_mean(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    assert not data.frechet_mean() is None

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_mean_single_item(gen_data):
    M, y = gen_data(1)
    data = MetricData(M, y)
    assert M.d(data.frechet_mean(), data[0]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_mean_weighted(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    weights = np.ones(10) / 10
    assert not data.frechet_mean(weights) is None

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_mean_single_weighted(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    weights = np.zeros(10)
    weights[5] = 1.0
    assert M.d(data.frechet_mean(weights), data[5]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_medoid(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    assert not data.frechet_medoid(n_jobs=None) is None

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_medoid_single_item(gen_data):
    M, y = gen_data(1)
    data = MetricData(M, y)
    assert M.d(data.frechet_medoid(n_jobs=None), data[0]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_medoid_single_weighted(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    weights = np.zeros(10)
    weights[5] = 1.0
    assert M.d(data.frechet_medoid(weights, n_jobs=None), data[5]) < TOL

@pytest.mark.parametrize("gen_data", GENERATORS)
def test_frechet_medoid_from_sample(gen_data):
    M, y = gen_data(10)
    data = MetricData(M, y)
    assert np.sum(M.d(data.frechet_medoid(n_jobs=None), data) < TOL) == 1

if __name__ == "__main__":
    pytest.main()