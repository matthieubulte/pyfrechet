from dataclasses import dataclass
from typing import Generator, Optional

from sklearn.cluster import KMeans

from pyfrechet.metric_spaces import MetricData
from pyfrechet.metric_spaces.utils import *
from .weighting_regressor import WeightingRegressor


@dataclass
class HonestIndices:
    fit_idx: np.ndarray
    predict_idx: np.ndarray


@dataclass
class Split:
    feature_idx: int
    threshold: float
    impurity: float


@dataclass
class Node:
    selector: HonestIndices
    split: Optional[Split]
    left: Optional['Node']
    right: Optional['Node']


def _2means_propose_splits(X_j):
    kmeans = KMeans(
        n_clusters=2,
        random_state=0, 
        n_init=1
    ).fit(X_j.reshape((X_j.shape[0], 1)))
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
        super().__init__(precompute_distances=(impurity_method == 'medoid'))
        
        # TODO: parameter constraints, see https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_forest.py#L199
        self.min_split_size = min_split_size
        self.impurity_method = impurity_method
        self.split_type = split_type
        self.is_honest = is_honest
        self.honesty_fraction = honesty_fraction
        self.root_node = None

    def _var(self, y):
        if self.impurity_method == 'cart':
            return y.frechet_var()
        elif self.impurity_method == 'medoid':
            return y.frechet_medoid_var()
        else:
            raise NotImplementedError(f'impurity_method = {self.impurity_method}')
 
    def _propose_splits(self, Xj) -> Generator[float, None, None]:
        if self.split_type == 'greedy':
            return _greedy_propose_splits(Xj)
        elif self.split_type == '2means':
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

    def _split_to_idx(self, X, node):
        split = node.split
        sel = node.selector

        # fit part
        left_idx_fit = sel.fit_idx[X[sel.fit_idx, split.feature_idx] < split.threshold]
        right_idx_fit = sel.fit_idx[X[sel.fit_idx, split.feature_idx] >= split.threshold]

        # predict part
        left_idx_pred = sel.predict_idx[X[sel.predict_idx, split.feature_idx] < split.threshold]
        right_idx_pred = sel.predict_idx[X[sel.predict_idx, split.feature_idx] >= split.threshold]
        
        # merge back into HonestIndices
        return (HonestIndices(left_idx_fit, left_idx_pred), HonestIndices(right_idx_fit, right_idx_pred))

    def _init_idx(self, N):
        if self.is_honest:
            s = int(self.honesty_fraction * N)
            perm = np.random.permutation(N)
            return HonestIndices(perm[:s], perm[s:])
        else:
            all_idx = np.arange(N)
            return HonestIndices(all_idx, all_idx)

    def fit(self, X, y: MetricData):
        super().fit(X, y)

        N = X.shape[0]
        
        root = Node(self._init_idx(N), None, None, None)
        self.root_node = root
        queue = [root]
        while queue:
            node = queue.pop(0)
            split = self._find_split(X[node.selector.fit_idx, :], y[node.selector.fit_idx])
            if split:
                node.split = split
                left_indices, right_indices = self._split_to_idx(X, node)

                node.left = Node(left_indices, None, None, None)
                node.right = Node(right_indices, None, None, None)
                queue.append(node.left)
                queue.append(node.right)
                # free up space by removing selectors not needed in the nodes
                node.selector = None

        return self

    def _selector_to_weights(self, selector):
        weights = np.zeros(self.y_train_.shape[0])
        weights[selector] = 1.0
        return weights

    def weights_for(self, x):
        assert self.root_node
        node = self.root_node
        while True and node:
            if not node.split:
                return self._normalize_weights(self._selector_to_weights(node.selector.predict_idx), sum_to_one=True, clip=True)
            elif x[node.split.feature_idx] < node.split.threshold:
                node = node.left
            else:
                node = node.right