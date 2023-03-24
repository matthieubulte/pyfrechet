from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import Any, Union
import numpy as np
from distance_utils import *
from sklearn.cluster import KMeans
from tqdm import tqdm

@dataclass
class Split:
    feature_idx: int
    threshold: float
    impurity: float

@dataclass
class Node:
    selector: Any
    split: Union[Split, None]
    left: Any
    right: Any
        
def print_node(node, ntabs=0):
    n = np.sum(node.selector)
    tabs = '  ' * ntabs
    if node.left:
        print(f'{tabs}Node(n={n}, split_idx={node.split.feature_idx}, val={node.split.threshold:.2f})')
        print_node(node.left, ntabs=ntabs+1)
        print_node(node.right, ntabs=ntabs+1)
    else:
        print(f'{tabs}Leaf(n={n})')

def __find_split(x, D, min_split_size = 5):
    N = x.shape[0]
    d = x.shape[1]
    
    split_imp = np.inf
    split_j = None
    split_val = None
    
    for j in range(d):
        for i in range(N):
            sel = x[:,j] < x[i,j]
            n_l = sel.sum() 
            n_r = (~sel).sum()

            if min(n_l, n_r) > min_split_size:
                var_l = medoid_var(mat_sel(D, sel))
                var_r = medoid_var(mat_sel(D, ~sel))
                impurity = (n_l * var_l + n_r * var_r)/N
                if impurity < split_imp:
                    split_imp = impurity
                    split_j = j
                    split_val = x[i,j]
                    
    return None if split_val is None else Split(split_j, split_val, split_imp)

def __find_split_std(x, y, frechet_var, min_split_size=5):
    N = x.shape[0]
    d = x.shape[1]
    
    split_imp = np.inf
    split_j = None
    split_val = None
    
    for j in range(d):
        for i in range(N):
            sel = x[:,j] < x[i,j]
            n_l = sel.sum() 
            n_r = (~sel).sum()

            if min(n_l, n_r) > min_split_size:
                var_r = frechet_var(y[~sel, :])
                var_l = frechet_var(y[sel, :])
                impurity = (n_l * var_l + n_r * var_r)/N
                if impurity < split_imp:
                    split_imp = impurity
                    split_j = j
                    split_val = x[i,j]
                    
    return None if split_val is None else Split(split_j, split_val, split_imp)

def __find_split_kmeans(x, y, frechet_var, min_split_size=5):
    N = x.shape[0]
    d = x.shape[1]
    
    split_imp = np.inf
    split_j = None
    split_val = None
    
    for j in range(d):
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(x[:,[j]])
        sel = kmeans.labels_.astype(bool)
        n_r = (~sel).sum()
        n_l = sel.sum() 

        if min(n_l, n_r) > min_split_size:
            var_r = frechet_var(y[~sel, :])
            var_l = frechet_var(y[sel, :])
            impurity = (n_l * var_l + n_r * var_r)/N
            if impurity < split_imp:
                split_imp = impurity
                split_j = j
                # 0 is the left cluster -> split is between max of cluster 0 and min of cluster 1
                if kmeans.cluster_centers_[0,0] < kmeans.cluster_centers_[1,0]:
                    split_val = (np.max(x[sel, j]) + np.min(x[~sel, j])) / 2
                else:
                    split_val = (np.min(x[sel, j]) + np.max(x[~sel, j])) / 2
                
                    
    return None if split_val is None else Split(split_j, split_val, split_imp)

def reg_tree(x, D, basemask=None):
    N = x.shape[0]
    if basemask is None:
        basemask = np.repeat(True, N)
    
    root = Node(basemask, None, None, None)
    queue = [ root ]
    while queue:
        node = queue.pop(0)
        split = __find_split(x[node.selector, :], mat_sel(D, node.selector))
        if split:
            split_selector = x[:, split.feature_idx] < split.threshold
            node.split = split
            node.left = Node(node.selector & split_selector, None, None, None)
            node.right = Node(node.selector & (~split_selector), None, None, None)
            queue.append(node.left)
            queue.append(node.right)
    return root

def __reg_tree_std(x, y, frechet_var, basemask=None):
    N = x.shape[0]
    if basemask is None:
        basemask = np.repeat(True, N)
    
    root = Node(basemask, None, None, None)
    queue = [ root ]
    while queue:
        node = queue.pop(0)
        split = __find_split_std(x[node.selector, :], y[node.selector, :], frechet_var)
        if split:
            split_selector = x[:, split.feature_idx] < split.threshold
            node.split = split
            node.left = Node(node.selector & split_selector, None, None, None)
            node.right = Node(node.selector & (~split_selector), None, None, None)
            queue.append(node.left)
            queue.append(node.right)
    return root

def __reg_tree_kmeans(x, y, frechet_var, basemask=None):
    N = x.shape[0]
    if basemask is None:
        basemask = np.repeat(True, N)
    
    root = Node(basemask, None, None, None)
    queue = [ root ]
    while queue:
        node = queue.pop(0)
        split = __find_split_kmeans(x[node.selector, :], y[node.selector, :], frechet_var)
        if split:
            split_selector = x[:, split.feature_idx] < split.threshold
            node.split = split
            node.left = Node(node.selector & split_selector, None, None, None)
            node.right = Node(node.selector & (~split_selector), None, None, None)
            queue.append(node.left)
            queue.append(node.right)
    print(basemask.sum(), i)
    return root

def tree_selector(node, x):
    while True:
        if not node.split:
            return node.selector
        elif x[node.split.feature_idx] < node.split.threshold:
            node = node.left
        else:
            node = node.right

def rf(n_estimators, x, D, s, with_tqdm=False):
    """
    Generate a random forest with specified number of trees.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest.
    
    x : numpy.ndarray
        The feature matrix with shape (n_samples, n_features).
    
    D : numpy.ndarray
        The distance matrix with shape (n_samples, n_samples).
    
    s : int
        The number of samples to draw for each tree.
    
    with_tqdm : bool, optional (default=False)
        Whether to display a progress bar using tqdm.

    Returns
    -------
    list
        A list of tree objects, each representing a tree in the forest.
    """
    N = x.shape[0]
    trees = []
    it = tqdm(range(n_estimators)) if with_tqdm else range(n_estimators)
    for i in it:
        mask = np.repeat(False, N)
        mask[np.random.choice(N,s,replace=False)] = True
        trees.append(reg_tree(x, D, basemask=mask))

    return trees

def rf_par(n_estimators, x, D, s, n_jobs=-2):
    """
    Generate a random forest with specified number of trees in parallel.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest.
    
    x : numpy.ndarray
        The feature matrix with shape (n_samples, n_features).
    
    D : numpy.ndarray
        The dissimilarity matrix with shape (n_samples, n_samples).
    
    s : int
        The number of samples to draw for each tree.
    
    n_jobs : int, optional (default=-2)
        The number of jobs to run in parallel for generating trees.
        -1 means using all processors, -2 means using all processors but one.

    Returns
    -------
    list
        A list of tree objects, each representing a tree in the forest.
    """
    N = x.shape[0]
    def make_mask():
        mask = np.repeat(False, N)
        mask[np.random.choice(N,s,replace=False)] = True
        return mask

    def calc(mask): return reg_tree(x, D, basemask=mask)
    return Parallel(n_jobs=n_jobs, verbose=1)(delayed(calc)(make_mask()) for _ in range(n_estimators))
    
def rf_std(n_estimators, x, y, frechet_var, s):
    N = x.shape[0]
    trees = []
    for i in range(n_estimators):
        mask = np.repeat(False, N)
        mask[np.random.choice(N,s,replace=False)] = True
        trees.append(__reg_tree_std(x, y, frechet_var, basemask=mask))
    return trees

def rf_kmeans(n_estimators, x, y, frechet_var, s):
    N = x.shape[0]
    trees = []
    for i in range(n_estimators):
        mask = np.repeat(False, N)
        mask[np.random.choice(N,s,replace=False)] = True
        trees.append(__reg_tree_kmeans(x, y, frechet_var, basemask=mask))
    return trees

def rf_weights(forest, x):
    """
    Compute the weights for each training sample based on the given forest and the prediction point.

    Parameters
    ----------
    forest : list
        A list of tree objects, each representing a tree in the random forest.
    
    x : numpy.ndarray
        The feature matrix with shape (1, n_features).

    Returns
    -------
    numpy.ndarray
        An array of weights with shape (n_samples,) for each sample in the training data feature matrix.
    """
    weights = forest[1].selector*0.0
    for tree in forest:
        tree_sel = tree_selector(tree,x)+0
        weights += tree_sel / np.sum(tree_sel)
    return weights / len(forest)