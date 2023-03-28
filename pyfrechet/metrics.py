from sklearn.metrics import make_scorer

def mean_squared_error(M):
    def _mse(y_true, y_pred): return (M.d(y_true, y_pred)**2).mean()
    return make_scorer(_mse, greater_is_better=False)