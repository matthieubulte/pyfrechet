from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generator

from sklearn.cluster import KMeans

from metric_spaces import MetricData
from metric_spaces.utils import *
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


class MedoidVarMixin:
    @staticmethod
    def _var(y: MetricData):
        return y.frechet_medoid_var()


class CartVarMixin:
    @staticmethod
    def _var(y: MetricData):
        return y.frechet_var()


class KMeansSplitMixin:
    @staticmethod
    def _propose_splits(X_j):
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_j.reshape((X_j.shape[0], 1)))
        assert kmeans.labels_
        sel = kmeans.labels_.astype(bool)
        if kmeans.cluster_centers_[0, 0] < kmeans.cluster_centers_[1, 0]:
            split_val = (np.max(X_j[sel]) + np.min(X_j[~sel])) / 2
        else:
            split_val = (np.min(X_j[sel]) + np.max(X_j[~sel])) / 2
        yield split_val


class GreedySplitMixin:
    @staticmethod
    def _propose_splits(X_j):
        for i in range(X_j.shape[0]):
            yield X_j[i]


class Tree(WeightingRegressor, metaclass=ABCMeta):
    def __init__(self, min_split_size=5):
        self.min_split_size = min_split_size
        self.root_node = None

    @abstractmethod
    def _var(self, y):
        pass

    @abstractmethod
    def _propose_splits(self, Xj) -> Generator[float, None, None]:
        pass

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

        if basemask is None:
            basemask = np.repeat(True, N)

        root = Node(basemask, None, None, None)
        self.root_node = root
        queue = [root]
        while queue:
            node = queue.pop(0)
            split = self._find_split(X[node.selector, :], y[node.selector])
            if split:
                split_selector = X[:, split.feature_idx] < split.threshold
                node.split = split
                node.left = Node(node.selector & split_selector, None, None, None)
                node.right = Node(node.selector & (~split_selector), None, None, None)
                queue.append(node.left)
                queue.append(node.right)
        return self

    def weights_for(self, x):
        assert self.root_node
        node = self.root_node
        while True:
            if not node.split:
                return self._normalize_weights(0.0+node.selector, sum_to_one=True, clip=True)
            elif x[node.split.feature_idx] < node.split.threshold:
                node = node.left
            else:
                node = node.right

    def clone(self):
        return type(self)(self.min_split_size)


class MedoidTree(MedoidVarMixin, GreedySplitMixin, Tree):
    pass


class CartTree(CartVarMixin, GreedySplitMixin, Tree):
    pass


class KMeansMedoidTree(MedoidVarMixin, KMeansSplitMixin, Tree):
    pass


class KMeansCartTree(CartVarMixin, KMeansSplitMixin, Tree):
    pass
