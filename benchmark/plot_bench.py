import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_forest_df(df, n_cores=20, n_trees=100, Ns=[100,200,400]):
    _df = df.groupby(['N', 'p', 'method']).mean().reset_index().copy()

    T = lambda method, N, p: _df[(_df.method == method) & (_df.N == N) & (_df.p == p)].fitting_duration.iloc[0]
    O = lambda method, N: _df[(_df.method == method) & (_df.N == N)].dist_duration.iloc[0]
    TF = lambda method, N, p, n_trees, n_cores: O(method, N) + (1.0*n_trees / n_cores) * T(method, int(N/2), p)

    rows = []
    for N in Ns:
        for p in _df.p.unique():
            for method in _df.method.unique():
                rows.append((method, N, p, TF(method, N, p, n_trees, n_cores)))

    return pd.DataFrame(rows, columns=['method', 'N', 'p', 'duration'])


def plot_forest_df(forest_df, ref_method=None):
    forest_df = forest_df.copy()
    grid = sns.FacetGrid(forest_df, col="N", hue="method", palette="tab20c",)

    if ref_method:
        ylabel = 'rel duration'

        for N in forest_df['N'].unique():
            forest_df.loc[forest_df['N'] == N, 'duration'] /= forest_df[(forest_df['N'] == N) & (forest_df['p'] == 2) & (forest_df.method == ref_method)].duration.iloc[0]
    else:
        ylabel = 'duration'

    # forest_df['duration'] = np.log10(forest_df['duration'])

    grid.map(plt.plot, "p", 'duration', marker="o")
    grid.axes[0][0].set_ylabel(ylabel)

    grid.add_legend()
    return grid

