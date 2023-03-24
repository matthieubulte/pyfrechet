from skfda import FDataGrid
from skfda.misc.operators import SRSF
from skfda.representation.interpolation import SplineInterpolation
from skfda.exploratory.stats._fisher_rao import _elastic_alignment_array,_fisher_rao_warping_mean
from skfda._utils import invert_warping
import scipy 
import numpy as np

def fisher_rao_karcher_mean(
    fdatagrid: FDataGrid,
    weights,
    center: bool = True,
    max_iter: int = 20,
    tol: float = 1e-3,
    grid_dim = 7
) -> FDataGrid:
    
    srsf_transformer = SRSF(initial_value=0)
    fdatagrid_srsf = srsf_transformer.fit_transform(fdatagrid)
    eval_points = fdatagrid.grid_points[0]
    interpolation = SplineInterpolation(interpolation_order=3, monotone=True)

    # Discretisation points
    fdatagrid_normalized = FDataGrid(
        fdatagrid(eval_points),
        grid_points=eval_points,
    )

    srsf = fdatagrid_srsf(eval_points)[..., 0]
    # Initialize with function closest to the L2 mean with the L2 distance
    centered = srsf - np.dot(weights, srsf)

    distances = scipy.integrate.simps(
        np.square(centered, out=centered),
        eval_points,
        axis=1,
    )

    # Initialization of iteration
    mu = srsf[np.argmin(distances)]
    mu_aux = np.empty(mu.shape)
    mu_1 = np.empty(mu.shape)

    # Main iteration
    for _ in range(max_iter):

        gammas_matrix = _elastic_alignment_array(
            mu,
            srsf,
            eval_points,
            0,
            grid_dim,
        )

        gammas = FDataGrid(
            gammas_matrix,
            grid_points=eval_points,
            interpolation=interpolation,
        )

        fdatagrid_normalized = fdatagrid_normalized.compose(gammas)
        srsf = srsf_transformer.transform(
            fdatagrid_normalized,
        ).data_matrix[..., 0]

        # Next iteration
        mu_1 = np.dot(weights, srsf, out=mu_1)

        # Convergence criterion
        mu_norm = np.sqrt(
            scipy.integrate.simps(
                np.square(mu, out=mu_aux),
                eval_points,
            ),
        )

        mu_diff = np.sqrt(
            scipy.integrate.simps(
                np.square(mu - mu_1, out=mu_aux),
                eval_points,
            ),
        )

        if mu_diff / mu_norm < tol:
            break

        mu = mu_1

    
    initial = fdatagrid.data_matrix[:, 0].mean()

    srsf_transformer.set_params(initial_value=initial)

    # Karcher mean orbit in space L2/Gamma
    karcher_mean = srsf_transformer.inverse_transform(
        fdatagrid.copy(
            data_matrix=[mu],
            grid_points=eval_points,
            sample_names=("Karcher mean",),
        ),
    )

    if center:
        # Gamma mean in Hilbert Sphere
        mean_normalized = _fisher_rao_warping_mean(gammas)

        gamma_mean = FDataGrid(
            mean_normalized.data_matrix[..., 0],
            grid_points=eval_points,
        )

        gamma_inverse = invert_warping(gamma_mean)

        karcher_mean = karcher_mean.compose(gamma_inverse)

    # Return center of the orbit
    return karcher_mean

