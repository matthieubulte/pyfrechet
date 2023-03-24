from skfda import FDataGrid
from skfda.misc.operators import SRSF
from skfda.representation.interpolation import SplineInterpolation
from skfda._utils import normalize_scale
import scipy 
import numpy as np

def fisher_rao_warping_mean(
    warping: FDataGrid,
    weights,
    max_iter: int = 100,
    tol: float = 1e-6,
    step_size: float = 0.3,
) -> FDataGrid:
    
    eval_points = warping.grid_points[0]
    original_eval_points = eval_points

    # Rescale warping to (0, 1)
    if warping.grid_points[0][0] != 0 or warping.grid_points[0][-1] != 1:

        eval_points = normalize_scale(eval_points)
        warping = FDataGrid(
            normalize_scale(warping.data_matrix[..., 0]),
            normalize_scale(warping.grid_points[0]),
        )

    # Compute srsf of warpings and their mean
    srsf = SRSF(output_points=eval_points, initial_value=0)
    psi = srsf.fit_transform(warping)
    psi_data = psi.data_matrix[..., 0]

    # Find psi closest to the mean
    # psi_centered_data = psi_data - np.dot(weights, psi_data)
    # np.square(psi_centered_data, out=psi_centered_data)
    # d = psi_centered_data.sum(axis=1).argmin()
    d = np.argmax(weights)

    # Get raw values to calculate
    mu = np.atleast_2d(psi[d].data_matrix[0, ..., 0])
    vmean = np.empty((1, len(eval_points)))

    # Construction of shooting vectors
    for _ in range(max_iter):

        vmean[0] = 0
        i = 0
        # Compute shooting vectors
        for psi_i in psi_data:
            inner = scipy.integrate.simps(mu * psi_i, x=eval_points)
            inner = max(min(inner, 1), -1)

            theta = np.arccos(inner)

            if theta > 1e-10:
                vmean += weights[i] * theta / np.sin(theta) * (psi_i - np.cos(theta) * mu)

            i += 1

        # Mean of shooting vectors
        v_norm = np.sqrt(scipy.integrate.simps(np.square(vmean)))

        # Convergence criterion
        if v_norm < tol:
            break

        # Calculate exponential map of mu
        a = np.cos(step_size * v_norm)
        b = np.sin(step_size * v_norm) / v_norm
        mu = a * mu + b * vmean

    # Recover mean in original gamma space
    warping_mean_ret = scipy.integrate.cumtrapz(
        np.square(mu, out=mu)[0],
        x=eval_points,
        initial=0,
    )

    # Affine traslation to original scale
    warping_mean_ret = normalize_scale(
        warping_mean_ret,
        a=original_eval_points[0],
        b=original_eval_points[-1],
    )

    monotone_interpolation = SplineInterpolation(
        interpolation_order=3,
        monotone=True,
    )

    return FDataGrid(
        [warping_mean_ret],
        grid_points=original_eval_points,
        interpolation=monotone_interpolation,
    )

