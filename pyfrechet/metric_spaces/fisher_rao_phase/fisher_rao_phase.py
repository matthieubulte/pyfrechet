import numpy as np
import fdasrsf
import scipy
import skfda
from skfda.misc.operators import SRSF
from skfda.exploratory.stats._fisher_rao import _elastic_alignment_array
from skfda._utils import normalize_scale, invert_warping
from ..metric_space import MetricSpace
from .fisher_rao_warping_mean import fisher_rao_warping_mean

class FisherRaoPhase(MetricSpace):
    def __init__(self, grid):
        self.grid = grid

    def _d(self, x, y):
        # elastic_distance returns (Dy: amplitude distance, Dx: phase distance)
        return fdasrsf.utility_functions.elastic_distance(x, y, self.grid)[1]
    
    def _find_template_srsf(self, fd_q):
        srsf_data = fd_q.data_matrix[...,0]
        eval_points_normalized = normalize_scale(self.grid)
        centered = (srsf_data.T - srsf_data.mean(axis=0, keepdims=True).T).T
        distances = scipy.integrate.simps(
            np.square(centered, out=centered),
            eval_points_normalized,
            axis=1,
        )

        # Initialization of iteration
        mu_idx = np.argmin(distances)
        mu_matrix = srsf_data[mu_idx]
        gammas_matrix = _elastic_alignment_array(mu_matrix, srsf_data, eval_points_normalized, 0, 10)

        return mu_idx, fd_q.copy(data_matrix=gammas_matrix, grid_points=self.grid)

    def _frechet_mean(self, y, w):
        # only keep observations with w > 0 for better initialization of the mean and perf
        sel = np.argwhere(w > 0)[...,0]
        y = skfda.FDataGrid(y[sel], self.grid)
        w = w[sel]
        
        fd_q = SRSF().fit_transform(y)
        template_idx, fd_y_warps = self._find_template_srsf(fd_q)
        gamma_mean = fisher_rao_warping_mean(fd_y_warps, w)
        gamma_inverse = invert_warping(gamma_mean)
        return y[template_idx].compose(gamma_inverse).data_matrix[0,:,0]

    def __str__(self):
        return 'FisherRaoPhase'
