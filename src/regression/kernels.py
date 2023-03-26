import numpy as np

def epanechnikov(u):
    return 3.0 * (1.0 - np.minimum(1, u**2))/4.0

def gaussian(u):
    return np.exp(-u**2/2) /np.sqrt(2*np.pi)