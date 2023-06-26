import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_forest_df(df, n_cores=20, n_trees=100, Ns=[100,200,400]):
    _df = df.groupby(['N', 'p', 'method']).mean().reset_index().copy()
    
    has_dist = lambda method: 1.0 * (method in ['medoid_2means', 'medoid_greedy'])

    # sometime it was medoid_2means and sometimes medoid_greedy used as a ref. Just take the one with the largest dist_duration
    # since it's the one which actually ocmputed it
    O = lambda method, N: has_dist(method) * _df[_df.N == N].dist_duration.max()

    T = lambda method, N, p: _df[(_df.method == method) & (_df.N == N) & (_df.p == p)].fitting_duration.iloc[0]
    
    TF = lambda method, N, p, n_trees, n_cores: O(method, N) + (1.0*n_trees / n_cores) * T(method, int(N/2), p)

    rows = []
    for N in Ns:
        for p in _df.p.unique():
            for method in _df.method.unique():
                # print(N,p,method, O(method, N))
                rows.append((method, N, p, TF(method, N, p, n_trees, n_cores)))

    return pd.DataFrame(rows, columns=['method', 'N', 'p', 'duration'])


def plot_forest_df(forest_df, ref_method=None):
    sns.set_style("whitegrid")
    sns.set_context("paper", rc={
        "axes.labelsize": 20,
        "xtick.labelsize": 20
    })   
    forest_df = forest_df.copy().rename(columns={ 'p': 'd' })

    forest_df['Method'] = forest_df.method.map({
        'cart_2means': 'RFWLCFR',
        'medoid_greedy': 'MRF',
        # 'medoid_2means': 'MRF2M'
    })

    grid = sns.FacetGrid(
        forest_df,
        col="N",
        hue="Method",
        hue_order=[
            'MRF', 
            # 'MRF2M', 
            'RFWLCFR'
        ])

    if ref_method:
        ylabel = 'relative duration'

        # for N in forest_df['N'].unique():
        #     forest_df.loc[forest_df['N'] == N, 'duration'] /= forest_df[(forest_df['N'] == N) & (forest_df['p'] == 2) & (forest_df.method == ref_method)].duration.iloc[0]

        for N in forest_df['N'].unique():
            for p in forest_df['p'].unique():
                sel = (forest_df['p'] == p) & (forest_df['N'] == N)
                forest_df.loc[sel, 'duration'] /= forest_df[sel & (forest_df.method == ref_method)].duration.iloc[0]
    else:
        ylabel = 'Duration (s)'

    # forest_df['duration'] = np.log10(forest_df['duration'])

    grid.map(plt.plot, "d", 'duration', marker="o")
    grid.axes[0][0].set_ylabel(ylabel)
    
    for ax in grid.axes[0]:
        ax.grid(axis='x')
        ax.set_xticks(forest_df['d'].unique())


    # grid.add_legend()

    # sns.move_legend(grid, "upper right", ncol=1, frameon=True, bbox_to_anchor=(1, 0.95))
    # sns.move_legend(
    #     grid, "lower center",
    #     bbox_to_anchor=(.5, -0.02), ncol=3, title=None, frameon=False,
    # )

    return grid


def plot_errors(df):
    df = df.copy().rename(columns={ 'err': 'MSE', 'p': 'd' })
    df['Method'] = df.method.map({
        'cart_2means': 'RFWLCFR',
        'medoid_greedy': 'MRF',
        # 'medoid_2means': 'MRF2M'
    })
    sns.set_style("whitegrid")
    sns.set_context("paper", rc={
        "axes.labelsize": 20,
        "xtick.labelsize": 20
    })   
    

    grid = sns.catplot(df,
        x='N', y='MSE',
        col="d",
        hue="Method",
        hue_order=[
            'MRF',
            # 'MRF2M',
            'RFWLCFR'
        ],
        kind='box')
    
    grid._legend.remove()
    # grid.add_legend()
    
    # sns.move_legend(
    #     grid, "lower center",
    #     bbox_to_anchor=(.5, -0.02), ncol=3, title=None, frameon=False,
    # )
