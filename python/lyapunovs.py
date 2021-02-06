import numpy as np
import scipy
import local_jacobian as lj

# attractor should be a 2D array, i.e. attractor shape should return a (x,y) tuple
# where both x and y are defined.
def lyapunovs(attractor: np.ndarray, theiler_window: int, return_convergence = False) -> np.array:
    # total number of points and a dimension of the problem.
    n, m = attractor.shape
    if n < 2*m+1:
        raise Exception("The attractor has to be a vertical matrix (n > 2*m+1)!")
    # k-d tree for quick search for neighboring points.
    # the last point is excluded because it doesn't have a successor and it's
    # impossible to estimate local jacobian in it.
    kd_tree = scipy.spatial.KDTree(attractor[:n-2])
    # initial values for the exponents are zeros.
    λ = np.zeros(m)
    res = [λ]
    # initial value the orthogonal deviation vectors.
    Y = np.identity(m)
    for t in range(0, n-1):
        # calculate the local expansion rate.
        J = lj.local_jacobian(t, attractor, kd_tree, theiler_window)
        # apply the local jacobian by the tangent vectors and reorthogonalize them.
        Q, R = np.linalg.qr(J.dot(Y))
        # take the logarithms of the new tangent vectors' norms and accumulate their values.
        # don't dother with computing eigs, because they are just the main diagonal elements.
        for i in range(0,m):
            λ[i] += np.log(abs(R[i, i]))
        # initialize the tangent vectors set with the new vectors of unit length.
        Y = Q
        if return_convergence:
            res = np.append(res, [λ / (t+1)], axis = 0)
    # normalize the exponents.
    if return_convergence:
        return res
    return λ / (n-1)
