# pyfréchet

*pyfréchet* is a Python module designed for the manipulation and analysis of data in metric spaces. It provides useful classes and methods for those looking to analyze non-standard data or develop new algorithms.

The package offers two essential building blocks for working with metric space-valued data: several implementations of metric spaces as subclasses of the `MetricSpace` class, and a dataframe-like class `MetricData` for holding a collection of metric space-valued data.

## Example

Currently, the package only implements regression methods with Euclidean predictors. Here's an example of how to use the package:

```python
from sklearn.model_selection import train_test_split

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.sphere import Sphere
from pyfrechet.regression.knn import KNearestNeighbours

M = Sphere(dim=1)

X, y = random_data(n=300) # Generate random covariates in R^p and responses on the unit circle S^1
y = MetricData(M, y) # Wrap the circle data in a MetricData object with the corresponding metric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # The MetricData class is implemented with compatibility in mind, allowing interaction with libraries from the Python ecosystem

# pyfréchet implements the widely-used scikit-learn API for fitting and evaluating models
knn = KNearestNeighbours(k=5)
knn.fit(X_train, y_train)
test_predictions = knn.predict(X_test)

test_error = M.d(y_test, test_predictions)
print(f'MSE = {(test_error**2).mean()}')
```

## Metric Spaces
The package supports the following metric spaces:
- Euclidean spaces $\mathbb{R}^d$
- Spheres $S^{d-1}$
- 1D Wasserstein spaces with the $L_2$ distance
- Functions equipped with the Fisher-Rao Phase distance
- Correlation matrices equipped with the Frobenius distance

To add support for more metric spaces, simply create a subclass of the `MetricSpace` class and provide an implementation of the distance function and the weighted Fréchet mean in that space.

## Methods
The following regression methods are (partially) implemented:
- Global Fréchet regression
- Local Fréchet regression (only with $p=1$ predictor)
- Nadaraya-Watson 
- K Nearest Neighbors
- Random forest (with 4 different splitting schemes - 2x2)

⚠️ **This package is under heavy development, meaning some example notebooks might not be updated to match the rest of the codebase, and documentation and references to original sources may be missing** ⚠️