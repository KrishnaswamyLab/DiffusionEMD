""" Implements convolutional sinkhorn distances from Solomon et al. 2015
"""
import numpy as np
import pygsp


def conv_sinkhorn(
    W, m_0, m_1, stopThr=1e-4, max_iter=1e3, method="chebyshev", t=50, verbose=False
):
    """ Implements the convolutional sinkhorn operator described in Solomon et
    al. 2015. This is sinkhorn except the cost matrix is replaced with the heat
    operator which may be easier to apply.

    Notes: It is unclear how to pick t from the manuscript. We will pick by
    cross validation.

    Parameters
    ----------
    W, n x n adjacency matrix of a graph
    m_0, m_1 distributions over W numpy arrays of length n
    """
    eps = 1e-8
    N = W.shape[0]
    G = pygsp.graphs.Graph(W)
    if method == "chebyshev":
        G.estimate_lmax()
    elif method == "exact":
        G.compute_fourier_basis()
    else:
        raise NotImplementedError("Unknown method %s" % method)
    heat_filter = pygsp.filters.Heat(G, t)
    v = np.ones(N)
    w = np.ones(N)
    for i in range(1, int(max_iter) + 1):
        v_prev = v
        v = m_0 / (heat_filter.filter(w, method=method) + eps)
        w = m_1 / (heat_filter.filter(v, method=method) + eps)
        if i % 100 == 0:
            if verbose:
                print(i, np.sum(np.abs(v - v_prev)))
            if np.sum(np.abs(v - v_prev)) < stopThr:
                if verbose:
                    print("converged at iteration %d" % i)
                break

    return np.sum(t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps)))
