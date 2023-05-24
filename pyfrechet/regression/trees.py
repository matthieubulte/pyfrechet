from dataclasses import dataclass
from typing import Generator

from sklearn.cluster import KMeans

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


@dataclass
class Split:
    feature_idx: int
    threshold: float
    impurity: float


@dataclass
class Node:
    selector: np.ndarray
    split: Split
    left: 'Node'
    right: 'Node'


def _2means_propose_splits(X_j):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_j.reshape((X_j.shape[0], 1)))
    assert not kmeans.labels_ is None
    sel = kmeans.labels_.astype(bool)
    if kmeans.cluster_centers_[0, 0] < kmeans.cluster_centers_[1, 0]:
        split_val = (np.max(X_j[sel]) + np.min(X_j[~sel])) / 2
    else:
        split_val = (np.min(X_j[sel]) + np.max(X_j[~sel])) / 2
    yield split_val


def _greedy_propose_splits(X_j):
    for i in range(X_j.shape[0]):
        yield X_j[i]


class Tree(WeightingRegressor):
    def __init__(self, 
                 split_type='greedy',
                 impurity_method='cart',
                 min_split_size=5,
                 is_honest=False,
                 honesty_fraction=0.5):
        super().__init__(precompute_distances=impurity_method is 'medoid')
        
        # TODO: parameter constraints, see https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_forest.py#L199
        self.min_split_size = min_split_size
        self.impurity_method = impurity_method
        self.split_type = split_type
        self.is_honest = is_honest
        self.honesty_fraction = honesty_fraction
        self.root_node = None

    def _var(self, y):
        if self.impurity_method is 'cart':
            return y.frechet_medoid_var()
        elif self.impurity_method is 'medoid':
            return y.frechet_var()
        else:
            raise NotImplementedError(f'impurity_method = {self.impurity_method}')
 
    def _propose_splits(self, Xj) -> Generator[float, None, None]:
        if self.split_type is 'greedy':
            return _greedy_propose_splits(Xj)
        elif self.split_type is '2means':
            return _2means_propose_splits(Xj)
        else:
            raise NotImplementedError(f'split_type = {self.split_type}')

    def _find_split(self, X, y: MetricData):
        N, d = X.shape

        split_imp = np.inf
        split_j = 0
        split_val = 0

        for j in range(d):
            for candidate_split_val in self._propose_splits(X[:, j]):
                sel = X[:, j] < candidate_split_val
                n_l = sel.sum()
                n_r = (~sel).sum()

                if min(n_l, n_r) > self.min_split_size:
                    var_l = self._var(y[sel])
                    var_r = self._var(y[~sel])
                    impurity = (n_l * var_l + n_r * var_r) / N
                    if impurity < split_imp:
                        split_imp = impurity
                        split_j = j
                        split_val = candidate_split_val

        return None if split_imp is np.inf else Split(split_j, split_val, split_imp)

    def fit(self, X, y: MetricData, basemask=None):
        super().fit(X, y)

        N = X.shape[0]
        # self.train_mask = np.repeat(True, N)

        # if self.is_honest:
            
        #     n_train_set = int(self.honesty_fraction * X.shape[0])
        #     self.train_mask = 


        if basemask is None:
            basemask = np.arange(N)

        root = Node(basemask, None, None, None)
        self.root_node = root
        queue = [root]
        while queue:
            node = queue.pop(0)
            split = self._find_split(X[node.selector, :], y[node.selector])
            if split:
                left_selector = node.selector[X[node.selector, split.feature_idx] < split.threshold]
                right_selector = node.selector[X[node.selector, split.feature_idx] >= split.threshold]

                node.split = split
                node.left = Node(left_selector, None, None, None)
                node.right = Node(right_selector, None, None, None)
                queue.append(node.left)
                queue.append(node.right)
                # free up space by removing selectors not needed in the nodes
                node.selector = None

        return self
    
    def cleanup_selectors(self):
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            if node and node.split:
                node.selector = None
                queue.append(node.left)
                queue.append(node.right)

    def _selector_to_weights(self, selector):
        weights = np.zeros(self.y_train_.shape[0])
        weights[selector] = 1.0
        return weights

    def weights_for(self, x):
        assert self.root_node
        node = self.root_node
        while True:
            if not node.split:
                return self._normalize_weights(self._selector_to_weights(node.selector), sum_to_one=True, clip=True)
            elif x[node.split.feature_idx] < node.split.threshold:
                node = node.left
            else:
                node = node.right