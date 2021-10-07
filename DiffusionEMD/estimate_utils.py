"""
Adapated from Vertex frequency codebase. Credit to Gabriel Dolsten.
Algorithms based on https://arxiv.org/pdf/1905.09758.pdf
Goal is to estimate the density of eigenvalues over a known range.
"""

import numpy as np
import scipy.sparse as ss
import numpy.random as nr
import matplotlib.pyplot as plt
import pygsp
import ot


def moments_cheb_dos(A, n, nZ=100, N=10, kind=1):
    """
    Compute a column vector of Chebyshev moments of the form c(k) = tr(T_k(A))
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should
    already lie in [-1,1]. The traces are computed via a stochastic estimator
    with nZ probe

    Args:
        A: Matrix or function apply matrix (to multiple RHS)
        n: Dimension of the space
        nZ: Number of probe vectors with which we compute moments
        N: Number of moments to compute
        kind: 1 or 2 for first or second kind Chebyshev functions
                (default = 1)

    Output:
        c: a column vector of N moment estimates
        cs: standard deviation of the moment estimator
                (std/sqrt(nZ))
    """

    # Create a function handle if given a matrix
    if callable(A):
        Afun = A
    else:
        if isinstance(A, np.ndarray):
            A = ss.csr_matrix(A)

        def Afun(x):
            return A * x

    if N < 2:
        N = 2

    # Set up random probe vectors (allowed to be passed in)
    if not isinstance(nZ, int):
        Z = nZ
        nZ = Z.shape[1]
    else:
        Z = np.sign(nr.randn(n, nZ))

    # Estimate moments for each probe vector
    cZ = moments_cheb(Afun, Z, N, kind)
    c = np.mean(cZ, 1)
    cs = np.std(cZ, 1, ddof=1) / np.sqrt(nZ)

    c = c.reshape([N, -1])
    cs = cs.reshape([N, -1])
    return c, cs


def moments_cheb(A, V, N=10, kind=1):
    """
    Compute a column vector of Chebyshev moments of the form c(k) = v'*T_k(A)*v
    for k = 0 to N-1. This routine does no scaling; the spectrum of A should
    already lie in [-1,1]

    Args:
        A: Matrix or function apply matrix (to multiple RHS)
        V: Starting vectors
        N: Number of moments to compute
        kind: 1 or 2 for first or second kind Chebyshev functions
                (default = 1)

    Output:
        c: a length N vector of moments
    """

    if N < 2:
        N = 2

    if not isinstance(V, np.ndarray):
        V = V.toarray()

    # Create a function handle if given a matrix
    if callable(A):
        Afun = A
    else:
        if isinstance(A, np.ndarray):
            A = ss.csr_matrix(A)

        def Afun(x):
            return A * x

    n, p = V.shape
    c = np.zeros((N, p))

    # Run three-term recurrence to compute moments
    TVp = V  # x
    TVk = kind * Afun(V)  # Ax
    c[0] = np.sum(V * TVp, 0)  # xx
    c[1] = np.sum(V * TVk, 0)  # xAx
    for i in range(2, N):
        TV = 2 * Afun(TVk) - TVp  # A*2T_1 - T_o
        TVp = TVk
        TVk = TV
        c[i] = sum(V * TVk, 0)
    return c


def plot_cheb_argparse(npts, c, xx0=-1, ab=np.array([1, 0])):
    """
    Handle argument parsing for plotting routines. Should not be called directly
    by users.

    Args:
        npts: Number of points in a default mesh
        c: Vector of moments
        xx0: Input sampling mesh (original coordinates)
        ab: Scaling map parameters

    Output:
        c: Vector of moments
        xx: Input sampling mesh ([-1,1] coordinates)
        xx0: Input sampling mesh (original coordinates)
        ab: Scaling map parameters
    """

    if isinstance(xx0, int):
        # only c is given
        xx0 = np.linspace(-1 + 1e-8, 1 - 1e-8, npts)
        xx = xx0
    else:
        if len(xx0) == 2:
            # parameters are c, ab
            ab = xx0
            xx = np.linspace(-1 + 1e-8, 1 - 1e-8, npts)
            xx0 = ab[0] * xx + ab[1]
        else:
            # parameteres are c, xx0
            xx = xx0

    # All parameters specified
    if not (ab == [1, 0]).all():
        xx = (xx0 - ab[1]) / ab[0]

    return c, xx, xx0, ab


def plot_chebint(varargin, npts=1001, pflag=True):
    """
    Given a (filtered) set of first-kind Chebyshev moments, compute the integral
    of the density:
            int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
    Output a plot of cumulative density function by default.

    Args:
        c: Array of Chebyshev moments (on [-1,1])
        xx: Evaluation points (defaults to mesh of 1001 pts)
        ab: Mapping parameters (default to identity)
        pflag: Option to output the plot

    Output:
        yy: Estimated cumulative density up to each xx point
    """

    # Parse arguments
    c, xx, xx0, ab = plot_cheb_argparse(npts, *varargin)

    N = len(c)
    txx = np.arccos(xx)
    yy = c[0] * (txx - np.pi) / 2
    for idx in np.arange(1, N):
        yy += c[idx] * np.sin(idx * txx) / idx

    yy *= -2 / np.pi

    # Plot by default
    if pflag:
        plt.plot(xx0, yy)
        # plt.ion()
        plt.show()
        # plt.pause(1)
        # plt.clf()

    return [xx0, yy]


def plot_chebhist(varargin, pflag=True, npts=21):
    """
    Given a (filtered) set of first-kind Chebyshev moments, compute the integral
    of the density:
        int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
    Output a histogram of cumulative density function by default.

    Args:
        c: Vector of Chebyshev moments (on [-1,1])
        xx: Evaluation points (defaults to mesh of 21 pts)
        ab: Mapping parameters (default to identity)
        pflag: Option to output the plot

    Output:
        yy: Estimated counts on buckets between xx points
    """

    # Parse arguments
    c, xx, xx0, ab = plot_cheb_argparse(npts, *varargin)

    # Compute CDF and bin the difference
    yy = plot_chebint((c, xx0, ab), pflag=False)
    yy = yy[1:] - yy[:-1]
    xm = (xx0[1:] + xx0[:-1]) / 2

    # Plot by default
    if pflag:
        plt.bar(xm + 1, yy, align="center", width=0.1)
        # plt.ion()
        plt.show()
        # plt.pause(1)
        # plt.clf()

    return [xm + 1, yy]


def matrix_normalize(W, mode="s"):
    """
    Normalize an adjacency matrix.

    Args:
        W: weighted adjacency matrix
        mode: string indicating the style of normalization;
            's': Symmetric scaling by the degree (default)
            'r': Normalize to row-stochastic
            'c': Normalize to col-stochastic

    Output:
        N: a normalized adjacency matrix or stochastic matrix (in sparse form)
    """

    dc = np.asarray(W.sum(0)).squeeze()
    dr = np.asarray(W.sum(1)).squeeze()
    [i, j, wij] = ss.find(W)

    # Normalize in desired style
    if mode in "sl":
        wij = wij / np.sqrt(dr[i] * dc[j])
    elif mode == "r":
        wij = wij / dr[i]
    elif mode == "c":
        wij = wij / dc[j]
    else:
        raise ValueError("Unknown mode!")

    N = ss.csr_matrix((wij, (i, j)), shape=W.shape)
    return N


def simple_diffusion_embeddings(graph, distribution_labels, subsample=False, scales=7):
    """ The plain version, without any frills.
    Return the vectors whose L1 distances are the EMD between the given distributions.
    The graph supplied (a PyGSP graph) should encompass both distributions.
    The distributions themselves should be one-hot encoded with the
    distribution_labels parameter.
    """
    heat_filter = pygsp.filters.Heat(
        graph, tau=[2 ** i for i in range(1, scales + 1)], normalize=False
    )
    diffusions = heat_filter.filter(distribution_labels, method="chebyshev", order=32)
    print(diffusions.shape)
    if subsample:
        rng = np.random.default_rng(42)
    if len(diffusions.shape) == 2:
        n_samples = 1
        n, n_scales = diffusions.shape
    else:
        n, n_samples, n_scales = diffusions.shape
    embeddings = []
    for i in range(n_scales):
        d = diffusions[..., i]
        weight = 0.5 ** (n_scales - i)
        if subsample:
            subsample_idx = rng.integers(n, size=n // 10)
            lvl_embed = weight * d[subsample_idx].T
        else:
            lvl_embed = weight * d.T
        embeddings.append(lvl_embed)
    if len(diffusions.shape) == 2:
        embeddings = np.concatenate(embeddings)
    else:
        embeddings = np.concatenate(embeddings, axis=1)
    return embeddings


def l1_distance_matrix(embeddings):
    """
    Gives a square distance matrix with the L1 distances between the provided embeddings
    """
    D = np.zeros((len(embeddings), len(embeddings)))
    for i, embed1 in enumerate(embeddings):
        for j, embed2 in enumerate(embeddings):
            D[i][j] = np.sum(np.abs(embed1 - embed2))
            D[j][i] = D[i][j]
    return D


def exact_ot(signals, dists):
    D = np.zeros((len(signals), len(signals)))
    for i, sig1 in enumerate(signals):
        for j, sig2 in enumerate(signals):
            sig1 = sig1.copy(order="C")
            sig2 = sig2.copy(order="C")
            dists = dists.copy(order="C")
            D[i][j] = ot.emd2(sig1, sig2, dists, processes=-2)
            D[j][i] = D[i][j]
    return D


def permutation_vector_to_matrix(E):
    """Convert a permutation vector E (list or rank-1 array, length n) to a
    permutation matrix (n by n).  The result is returned as a
    scipy.sparse.coo_matrix, where the entries at (E[k], k) are 1.
    """
    n = len(E)
    j = np.arange(n)
    return ss.coo_matrix((np.ones(n), (E, j)), shape=(n, n))
