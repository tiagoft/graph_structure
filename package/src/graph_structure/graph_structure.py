import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph

def nearest_neighbors(x, k=None, self_is_neighbor=False, metric='minkowski', n_jobs=1):
    if type(k)==float:
        k = int(k * x.shape[0])
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


def mean_neighborhood_similarity_from_points(X, Y, k, n_jobs=1, metric='minkowski'):
    """
    This is $NNGS(X, Y, k)$
    """
    nx = nearest_neighbors(X, k=k, n_jobs=n_jobs, metric=metric)
    ny = nearest_neighbors(Y, k=k, n_jobs=n_jobs, metric=metric)
    return mean_neighborhood_similarity_from_neighborhood(nx, ny)

# def get_umap_kernel(sigma):
#     def umap_kernel(X):
#         D = pairwise_distances(X, metric='sqeuclidean')
#         D1 = D - np.min(D, axis=1, keepdims=True)
#         D1 = D1 / np.mean(D1, axis=1, keepdims=True)
#         D2 = D - np.min(D, axis=0, keepdims=True)
#         D2 = D2 / np.mean(D2, axis=0, keepdims=True)
#         D = np.minimum(D1, D2)

#         return np.exp(-D/(2*sigma**2))
#     return umap_kernel

def get_rbf_kernel(sigma):
    def rbf_kernel(X):
        return np.exp(-pairwise_distances(X, metric='sqeuclidean')/(2*sigma**2))
    return rbf_kernel

def estimate_sigma(X, alpha=0.8):
    """
    Estimate the sigma parameter for the RBF kernel in the CKA-RBF method.
    """
    # Compute the pairwise distances
    D = pairwise_distances(X, X, metric='sqeuclidean')
    D = np.sort(D, axis=1)
    D = D[:, 1:]  # Remove the self-distance

    # Compute the median distance
    median_distance = np.median(D)
    sigma = median_distance * alpha
    return sigma

def get_linear_kernel():
    def linear_kernel(X):
        return X @ X.T
    return linear_kernel


def cka(X, Y, kernel_X=get_linear_kernel(), kernel_Y=get_linear_kernel()):
    """
    CKA with a linear kernel as in:
    Similarity of Neural Network Representations Revisited,
    Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
    Proceedings of the 36th International Conference on Machine Learning,
    PMLR 97:3519-3529, 2019
    https://proceedings.mlr.press/v97/kornblith19a.html
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute the kernel matrices
    K = kernel_X(X)
    L = kernel_Y(Y)

    KH = K-np.mean(K, axis=1) # KH
    LH = L-np.mean(L, axis=1) # LH

    # Compute the CKA
    cka = np.trace(KH @ LH) / np.sqrt(np.trace(KH @ KH) * np.trace(LH @ LH))
    return cka
