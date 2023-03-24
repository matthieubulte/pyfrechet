import numpy as np

GRID = np.linspace(0, 1, 102)[1:-1]

def d(x, y):
    return np.sqrt(np.trapz((x - y)**2, GRID))

def fm(y, w):
    return np.dot(w, y)

def noise(J=5, grid=GRID):
    def _T(K): return grid if K == 0 else grid - np.sin(grid * np.pi * K) / (np.pi * np.abs(K))
    def _K(): return 2 * (np.random.binomial(1,0.5) - 1)*np.random.poisson(3)
    U = np.sort(np.random.uniform(size=J-1))
    T = np.array([ _T(_K()) for _ in range(J) ])
    return U[0] * T[0,:] + np.dot(U[1:] - U[:-1], T[1:-1, :]) + (1 - U[-1]) * T[-1,:]
