import numpy as np

# This code is taken from: https://www.statsmodels.org/dev/_modules/statsmodels/stats/correlation_tools.html#cov_nearest
# adding the entire statsmodels as a dependency just for this function would be an overkill.
# The implementation in https://github.com/mikecroucher/nearest_correlation didn't work well for some simple cases (input is already SPD)

def clip_evals(x, value=0.0):  # threshold=0, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < value)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return x_new, clipped


def corr_nearest(corr, threshold=1e-15, n_fact=100):
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError("matrix is not square")

    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)

    max_iterations = int(len(corr) * n_fact)

    for ii in range(max_iterations):
        x_adj = x_new - diff
        x_psd, clipped = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        print(f"No solution found in {max_iterations} iteration")

    return x_new