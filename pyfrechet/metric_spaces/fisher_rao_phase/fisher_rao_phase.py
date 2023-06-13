import numpy as np
import skfda
from scipy.integrate import trapz
from fdasrsf.utility_functions import f_to_srsf
from ..metric_space import MetricSpace
from .fisher_rao_warping_mean import fisher_rao_warping_mean

class FisherRaoPhase(MetricSpace):
    def __init__(self, grid):
        self.grid = grid

    def _d(self, gam_x, gam_y):
        psi_x = f_to_srsf(gam_x, self.grid)
        psi_y = f_to_srsf(gam_y, self.grid)
        theta = np.clip(trapz(psi_x * psi_y, self.grid), -1, 1)
        return np.arccos(theta)
    
    def _frechet_mean(self, gammas, w):
        # only keep observations with w > 0 for better initialization of the mean and perf
        sel = np.argwhere(w > 0)[...,0]
        gammas = skfda.FDataGrid(gammas[sel], self.grid)
        w = w[sel]
        return fisher_rao_warping_mean(gammas, w).data_matrix[0,:,0]
        
    def __str__(self):
        return 'FisherRaoPhase'
