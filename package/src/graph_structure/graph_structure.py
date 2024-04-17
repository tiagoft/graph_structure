import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def nearest_neighbors(x, self_is_neighbor=False, metric='cosine'):
    D = pairwise_distances(x, x, metric=metric)
    if self_is_neighbor == False:
        np.fill_diagonal(D, np.inf)
    closest = np.argsort(D, axis=1)
    return closest


def _compute_jaccard_similarity(sx, sy):
    """
    Compute Jaccard similarity between two sets of indices.
    """
    return len(sx.intersection(sy)) / len(sx.union(sy))


def _mean_neighborhood_similarity(nx, ny, k):
    num_points = nx.shape[0]
    inter = 0
    for i in range(num_points):
        sx = set(nx[i, 0:k])
        sy = set(ny[i, 0:k])
        inter += _compute_jaccard_similarity(sx, sy)
    inter /= num_points
    return inter


def _mean_neighborhood_distance(nx, ny, k):
    return 1 - _mean_neighborhood_similarity(nx, ny, k)


def mean_neighborhood_similarity(A, B, k):
    """
    This is $D_g(A, B, k)$
    """
    nx = nearest_neighbors(A)
    ny = nearest_neighbors(B)
    return _mean_neighborhood_similarity(nx, ny, k)


def mean_neighborhood_distance(A, B, k):
    """
    This is $D_g(A, B, k)$
    """
    nx = nearest_neighbors(A)
    ny = nearest_neighbors(B)
    return _mean_neighborhood_distance(nx, ny, k)


def _mean_structural_distance(nx, ny):
    num_points = nx.shape[0]
    k_vals = list(range(1, num_points - 1))
    ns = np.array([_mean_neighborhood_distance(nx, ny, k) for k in k_vals])
    msd = np.max(1 - ns)
    return msd


def mean_structural_distance(A, B):
    """
    This is $S_g(A, B)$
    """
    nx = nearest_neighbors(A)
    ny = nearest_neighbors(B)
    return _mean_structural_distance(nx, ny)