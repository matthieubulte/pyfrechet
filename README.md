# py-fréchet

*py-fréchet* is a Python module for manipulation and analysis of data in metric spaces. It provides useful classes and methods that can be useful for people trying to analyze non-standard data as well as people wanting to develop new algorithms.

The package provides two basic building blocks to work with metric space-valued data: several implementations of metric spaces as subclasses of the `MetricSpace` class, and a dataframe-like class `MetricData` holding a collection of metric space-valued data.

## Example

At the moment, the package only implement regression methods with Euclidian predictor. Here is an example of how to use the package

```python
from sklearn.model_selection import train_test_split

from metric_spaces import MetricData
from metric_spaces.sphere import Sphere
from regression.knn import KNearestNeighbours

M = Sphere(dim=1)

X, y = random_data(n=300) # Generate random covariates in R^p and responses on the unit circle S^1
y = MetricData(M, y) # Wrap the circle data in a MetricData object with the corresponding metric

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # The MetricData class is implemented with compatibility in mind, meaning that you can interact with libraries from the python ecosystem

# py-fréchet implements the now standard scikit-learn api for fitting and evaluating models
knn = KNearestNeighbours(k=5)
knn.fit(X_train, y_train)
test_predictions = knn.predict(X_test)

test_error = M.d(y_test, test_predictions)
print(f'MSE = {(test_error**2).mean()}')
```


## Metric Spaces
The package supports the following metric spaces:
- Euclidian spaces $\mathbb{R}^d$
- Spheres $S^{d-1}$ (currently, only for $d=1$)
- 1D Wasserstein spaces with the $L_2$ distance
- Functions equipped with the Fisher-Rao Phase distance
- Correlation matrices equiped with the Frobenius distance

Support for more metric spaces can be easily added by create a subclass of the class `MetricSpace`, which requires providing an implementation of the distance function and of the weighted Fréchet mean in that space.

## Methods
The following regression methods are (partially) implemented:
- Global Fréchet regression
- Local Fréchet regression (only with $p=1$ predictor)
- Nadaraya-Watson 
- K Nearest Neighbors
- Random forest (with 2x2=4 different splitting schemes)

⚠️ **This package is under heavy developement, meaning that some example notebooks might not be adapted to the rest of the codebase, documentation and references to original sources are missing** ⚠️