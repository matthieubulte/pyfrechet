if __name__ == '__main__':
    import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

    import numpy as np
    from scipy import stats
    from benchmark import bench
    from pyfrechet.metric_spaces import *

    OUT_FILE = 'results/result_sphere_cross_est_new_dgp_test.json'

    def gen_data(N, p, alpha, beta, eps=0.1):
        M = Sphere(2)

        def m(x):
            eta = 2 * (x - 0.5).dot(beta) / np.sqrt(p) + alpha
            z = stats.logistic.cdf(eta)
            dz = np.sqrt(1 - z**2)
            pz = np.pi * z
            y= np.c_[dz * np.cos(pz), dz * np.sin(pz), z]
            return y

        def add_noise(x, sig):
            v = sig * np.random.normal(size=(M.manifold.dim+1,))
            pvx = v - np.dot(x, v) * x
            return M.manifold.metric.exp(pvx, x)

        x = np.random.rand(N*p).reshape((N,p))
        mx = m(x)
        y = np.array([ add_noise(mx[i,:], eps) for i in range(N)])
        return x, MetricData(M, y), MetricData(M, mx)

    from pyfrechet.metric_spaces import *
    from pyfrechet.regression.trees import Tree
    from pyfrechet.metrics import mse
    from datetime import datetime


    cart_2means = Tree(impurity_method='cart', split_type='2means')
    medoid_greedy = Tree(impurity_method='medoid', split_type='greedy')

    results = []
    for N in [400]:
        for p in [20]:
            for i in range(1):
                print(f'[{str(datetime.now())}] Progress: N={N}\tp={p}\ti={i}')
                beta = np.random.randn(p)
                alpha = np.random.randn()
                X_train, y_train, _ = gen_data(N, p, alpha, beta)
                X_test, _, mx_test = gen_data(50, p, alpha, beta)
                y_train.compute_distances()
                # cart_2means = cart_2means.fit(X_train, y_train)
                medoid_greedy = medoid_greedy.fit(X_train, y_train)
