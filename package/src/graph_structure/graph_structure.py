import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def nearest_neighbors(x, k=None, self_is_neighbor=False, metric='minkowski'):
    G = kneighbors_graph(x, k, mode='connectivity', metric=metric, include_self=self_is_neighbor, n_jobs=-1)
    A = []
    for i in range(G.shape[0]):
        A.append(G.getrow(i).nonzero()[1])

    A=np.vstack(A)
    return A

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
