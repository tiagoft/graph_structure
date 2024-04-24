import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def nearest_neighbors(x, self_is_neighbor=False, metric='cosine'):
    D = pairwise_distances(x, x, metric=metric)
    if self_is_neighbor == False:
        np.fill_diagonal(D, np.inf)
    closest = np.argsort(D, axis=1)
    return closest


def compute_jaccard_similarity(sx, sy):
    """
    Compute Jaccard similarity between two sets of indices.
    """
    return len(sx.intersection(sy)) / len(sx.union(sy))


def mean_neighborhood_similarity_from_neighborhood(nx, ny, k):
    num_points = nx.shape[0]
    inter = 0
    for i in range(num_points):
        sx = set(nx[i, 0:k])
        sy = set(ny[i, 0:k])
        inter += compute_jaccard_similarity(sx, sy)
    inter /= num_points
    return inter


def mean_neighborhood_similarity_from_points(A, B, k):
    """
    This is $D_g(A, B, k)$
    """
    nx = nearest_neighbors(A)
    ny = nearest_neighbors(B)
    return mean_neighborhood_similarity_from_neighborhood(nx, ny, k)
