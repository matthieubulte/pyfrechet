import pandas as pd
import numpy as np
import json

PATH = '~/Documents/university/phd/datasets/hmm_paper_PTT_3965_POSIX__clean_grid_aligned.csv'

def load_data():
    read_nparray = lambda col: np.array(json.loads(col))
    standardize = lambda s: (s - s.mean()) / s.var()

    df = pd.read_csv(PATH)
    df.depth = df.depth.map(read_nparray)
    df['time'] = df.duration.map(lambda duration: np.linspace(0, duration, duration+1))
    df.grid_aligned_depth = df.grid_aligned_depth.map(read_nparray)
    df.date_start = pd.to_datetime(df.date_start)
    df.date_end = pd.to_datetime(df.date_end)
    df['bottom_level'] = df.max_depth * 0.75
    df['bottom_percentage'] = df.apply(lambda row: (row.depth > row.bottom_level).sum() / row.duration, axis=1)
    df['is_U'] = df.apply(lambda row: row.bottom_percentage >= 0.66, axis=1)
    df['max_depth'] = df.apply(lambda row: row['grid_aligned_depth'].max(), axis=1)
    df['grid_aligned_depth_norm'] = df.apply(lambda row: row['grid_aligned_depth'] / row['max_depth'], axis=1)
    df['max_depth_std'] = standardize(df.max_depth)
    df['time_since_last_dive_std'] = standardize(df.time_since_last_dive)
    df['duration_std'] = standardize(df.duration)
    return df


def split_df(df, Xs, y, Ntest=1000):
    pidx = np.random.permutation(df.shape[0])
    test_idx = pidx[:Ntest]
    X_test = df.iloc[test_idx][Xs].values
    y_test = df.iloc[test_idx][y].values

    train_idx = pidx[Ntest:]
    X_train = df.iloc[train_idx][Xs].values
    y_train = df.iloc[train_idx][y].values

    return X_test, y_test, X_train, y_train

def split_idx(N, Ntest=1000):
    pidx = np.random.permutation(N)
    return pidx[:Ntest], pidx[Ntest:]
