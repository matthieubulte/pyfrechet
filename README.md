# pyfréchet

*pyfréchet* is a Python module designed for the manipulation and analysis of data in metric spaces. It provides useful classes and methods for those looking to analyze non-standard data or develop new algorithms.

The package offers two essential building blocks for working with metric space-valued data: several implementations of metric spaces as subclasses of the `MetricSpace` class, and a dataframe-like class `MetricData` for holding a collection of metric space-valued data.

## Example

Currently, the package only implements regression methods with Euclidean predictors. Here's an example of how to use the package:

```python
from sklearn.model_selection import train_test_split

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metrics import mean_squared_error
from pyfrechet.metric_spaces.sphere import Sphere
from pyfrechet.regression.knn import KNearestNeighbours

M = Sphere(dim=3)
mse = mean_squared_error(M)

X, y = random_data(n=300) # Generate random covariates in R^p and responses on the unit sphere S^2
y = MetricData(M, y) # Wrap the circle data in a MetricData object with the corresponding metric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # The MetricData class is implemented with compatibility in mind, allowing to use it with other libraries from the Python ecosystem

# pyfréchet implements the widely-used scikit-learn estimator API for fitting and evaluating models
knn = KNearestNeighbours(n_neighbors=5).fit(X_train, y_train)
test_predictions = knn.predict(X_test)

print(f'MSE = {mse(y_test, test_predictions)}')

```

## Another example 

As mentionned above, every estimator is a subclass of the `BaseEstimator` from `scikit-learn`, giving access to many model selection and validation tools. Consider for instance the selection of a kernel and bandwidth in a kernel regression problem. This can be done by grid-search cross validation using `scikit-learn`'s `GridSearchCV`[^5]

```python
# Standard imports as above ...

from sklearn.model_selection import GridSearchCV
from pyfrechet.regression.kernels import NadarayaWatson, gaussian, epanechnikov

M = Sphere(dim=3)
mse = mean_squared_error(M)

X, y = random_data(n=300) # Again, generate random covariates in R^p and responses on the unit sphere S^2

# Define the possible parameter values over which to do the search
param_grid = {
    'base_kernel': [gaussian, epanechnikov],
    'bw': np.logspace(-1, 1, 20)
}

# Define the Nadaraya-Watson estimator with parameters searched by cross-validation over the grid defined above
est = GridSearchCV(
    estimator=NadarayaWatson(),
    param_grid=param_grid,
    scoring=mse
)
est.fit(X, y)

# This new estimator can be used as any parameters
est.predict(np.random.rand(p))

# And the best estimator can be extracted
est.best_estimator_
```

## Metric Spaces
The package supports the following metric spaces:
- Euclidean spaces $\mathbb{R}^d$
- Spheres $S^{d-1}$
- 1D Wasserstein spaces with the $L_2$ distance[^4]
- Functions equipped with the Fisher-Rao Phase distance[^2]
- Correlation matrices equipped with the Frobenius distance

To add support for more metric spaces, simply create a subclass of the `MetricSpace` class and provide an implementation of the distance function and the weighted Fréchet mean in that space.

## Methods
The following regression methods are (partially) implemented:
- Global Fréchet regression[^1]
- Local Fréchet regression[^1] (only with $p=1$ predictor)
- Nadaraya-Watson 
- K Nearest Neighbors
- Random forest[^3] (with 4 different splitting schemes - 2x2)

## Testing

After installing the dependencies, you can run the test suite from the root of the project with the `test` rule:
```
make test
```

## License
The package is licensed under the BSD 3-Clause License. A copy of the [license](https://github.com/matthieubulte/pyfrechet/blob/main/LICENSE) can be found along with the code.


⚠️ **This package is under heavy development, meaning some example notebooks might not be updated to match the rest of the codebase, and documentation and references to original sources may be missing** ⚠️

[^1]: Petersen, A. and Müller, H.-G. (2019). Fréchet regression for random objects with Euclidean predictors. The Annals of Statistics, 47(2), 691--719.

[^2]: Srivastava, A., & Klassen, E. P. (2016). Functional and shape data analysis (Vol. 1). New York: Springer.

[^3]: Qiu, R., Yu, Z., & Zhu, R. (2022). Random Forests Weighted Local Fréchet Regression with Theoretical Guarantee. arXiv preprint arXiv:2202.04912.

[^4]: Panaretos, V. M., & Zemel, Y. (2020). An invitation to statistics in Wasserstein space (p. 147). Springer Nature.

[^5]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html