import numpy as np
import scipy

def get_neighbors(t: np.array, neighbors_count: int, attractor: np.ndarray, tree: scipy.spatial.kdtree.KDTree, theiler_window: int) -> np.ndarray:
    neighbors = np.zeros(0)
    additional_neighbors = 1
    while neighbors.size < neighbors_count:
        _, neighbors = tree.query(attractor[t], k=neighbors_count + additional_neighbors)
        neighbors = neighbors[theiler_window < abs(neighbors - t)]
        additional_neighbors += neighbors_count - neighbors.size
    return neighbors[:neighbors_count]

def local_jacobian(t: int, attractor: np.ndarray, tree: scipy.spatial.kdtree.KDTree, theiler_window: int) -> np.ndarray:
    # attractor dimensions
    n, m = attractor.shape
    if n < 2*m+1:
        raise Exception("The attractor has to be a vertical matrix (n > 2*m+1)!")
    # find 2*m nearest neighbors and add the current point to the pool so that we
    # will have 2*m+1 points to work with. also, scipy does not support theiler
    # windows (minimum distance between indeces), so a crutch solution is used here.
    neighbors = np.append([t],get_neighbors(t, 2*m, attractor, tree, theiler_window))
    # matrix of neighbors (X) and their successors (Y)
    X = np.append(np.ones((2*m+1,1)), attractor[neighbors], axis = 1)
    Y = attractor[neighbors+1]
    # estimate the local jacobian
    j, _, _, _ = np.linalg.lstsq(X, Y[:,0], rcond=None)
    J = np.append(np.zeros((0,m)), [j[1:]], axis = 0)
    for i in range(1, m):
        j, _, _, _ = np.linalg.lstsq(X, Y[:,i], rcond=None)
        J = np.append(J, [j[1:]], axis = 0)
    return J