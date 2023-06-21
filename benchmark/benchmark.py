import numpy as np
from pyfrechet.metric_spaces import *
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse
from datetime import datetime

import time
import sklearn
import json

def bench_it(name, est, X_train, y_train, X_test, mx_test):
    N,p = X_train.shape
    fitted = sklearn.clone(est)
    
    t0 = time.time()

    print(f'[{str(datetime.now())}] Distances for {name}')

    distances_duration = 0
    if est.precompute_distances:
        tt0 = time.time()
        y_train.compute_distances()
        distances_duration = time.time() - tt0

    print(f'[{str(datetime.now())}] dt = {distances_duration}')
    print(f'[{str(datetime.now())}] Fitting for {name}')

    fitted = fitted.fit(X_train, y_train)
    total_duration = time.time() - t0
    
    print(f'[{str(datetime.now())}] dt = {total_duration - distances_duration}')
    print(f'[{str(datetime.now())}] MSE for {name}')

    e = mse(mx_test, fitted.predict(X_test))
    return (name, N, p, total_duration, distances_duration, e)


def bench(
        gen_data,
        out_file,
        ps=[2, 5, 10, 20],
        Ns=[50, 100, 200, 400],
        replicas=50,
    ):
    results = []
    for N in Ns:
        for p in ps:
            for i in range(replicas):
                cart_2means = Tree(impurity_method='cart', split_type='2means')
                # cart_greedy = Tree(impurity_method='cart', split_type='greedy')
                # medoid_2means = Tree(impurity_method='medoid', split_type='2means')
                medoid_greedy = Tree(impurity_method='medoid', split_type='greedy')
                
                print(f'[{str(datetime.now())}] Progress: N={N}\tp={p}\ti={i}')
                beta = np.random.randn(p)
                alpha = np.random.randn()
                X_train, y_train, _ = gen_data(N, p, alpha, beta)
                X_test, _, mx_test = gen_data(50, p, alpha, beta)

                results.append(bench_it('cart_2means', cart_2means, X_train, y_train, X_test, mx_test))
                # results.append(bench_it('cart_greedy', cart_greedy, X_train, y_train, X_test, y_test))
                # results.append(bench_it('medoid_2means', medoid_2means, X_train, y_train, X_test, mx_test))
                results.append(bench_it('medoid_greedy', medoid_greedy, X_train, y_train, X_test, mx_test))

                with open(out_file, 'w') as f:
                    json.dump(results, f)
