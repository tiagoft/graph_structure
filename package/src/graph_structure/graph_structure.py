import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def nearest_neighbors(x, k=None, self_is_neighbor=False, metric='minkowski', n_jobs=1):
    G = kneighbors_graph(x, k, mode='connectivity', metric=metric, include_self=self_is_neighbor, n_jobs=n_jobs)
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


def mean_neighborhood_similarity_from_neighborhood(nx, ny):
    num_points = nx.shape[0]
    inter = 0
    for i in range(num_points):
        sx = set(nx[i])
        sy = set(ny[i])
        inter += compute_jaccard_similarity(sx, sy)
    inter /= num_points
    return inter


def mean_neighborhood_similarity_from_points(A, B, k, n_jobs=1, metric='minkowski'):
    """
    This is $D_g(A, B, k)$
    """
    nx = nearest_neighbors(A, k=k, n_jobs=n_jobs, metric=metric)
    ny = nearest_neighbors(B, k=k, n_jobs=n_jobs, metric=metric)
    return mean_neighborhood_similarity_from_neighborhood(nx, ny)

def cka(X1, X2):
    """
    CKA with a linear kernel as in:
    Similarity of Neural Network Representations Revisited,
    Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
    Proceedings of the 36th International Conference on Machine Learning,
    PMLR 97:3519-3529, 2019
    https://proceedings.mlr.press/v97/kornblith19a.html
    """
    # Center the data
    X1 = X1 - X1.mean(axis=0)
    X2 = X2 - X2.mean(axis=0)

    # Compute the kernel matrices
    K1 = X1.T @ X1
    K2 = X2.T @ X2

    # Compute the squared Frobenius norms
    norm1 = np.linalg.norm(K1, 'fro')
    norm2 = np.linalg.norm(K2, 'fro')

    # Compute the CKA
    cka = np.linalg.norm(X2.T @ X1, 'fro')**2 / (norm1 * norm2)
    return cka