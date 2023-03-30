import pandas as pd
import numpy as np

from metric_spaces import MetricData, Wasserstein1D


def _prepare_capital_bikeshare(inpath, outpath):
    pd.read_parquet(inpath).groupby(by=['dteday']).aggregate(dict({
        'season': 'min',
        'holiday': 'min',
        'weekday': 'min',
        'workingday': 'min',
    #     'weathersit': 'avg', # what is this?
        'temp': 'mean',
    #     'atemp': 'avg', # what is this?
        'hum': 'mean',
        'windspeed': 'mean',
        'cnt': lambda cnt: np.quantile(cnt, Wasserstein1D.GRID).tolist()
    })).rename({ 'cnt': 'num_rides_hourly_quantiles' }, axis=1).to_parquet(outpath)


def load_capital_bikeshare(path):
    df = pd.read_parquet(path)
    X = df[['season','holiday','weekday','workingday','temp','hum','windspeed']]
    y = MetricData(Wasserstein1D(), np.c_[[ np.array(arr) for arr in df.num_rides_hourly_quantiles.values ]])
    return X, y
