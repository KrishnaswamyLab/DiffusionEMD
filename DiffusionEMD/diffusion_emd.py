""" These functions provide a way to quickly embed a set of distributions over
a graph into vectors where the L_1 distance between these embeded vectors
corresponds to the Wasserstein distance between distributions.
"""

import numpy as np
import pygsp
import scipy
from scipy.linalg import qr

# from scipy.linalg.interpolative import interp_decomp
import scipy.sparse

from . import estimate_utils


def estimate_dos(A, pflag=False, npts=1001):
    """ Estimate the density of states of the matrix A

    A should be a matrix of with eigenvalues in tha range [-1, 1].
    """
    c = estimate_utils.moments_cheb_dos(A, A.shape[0], N=50)[0]
    return estimate_utils.plot_chebint((c,), pflag=pflag, npts=npts)


def approximate_rank(A, thresh):
    """ Determines the rank relative to a threshold as defined in
    https://doi.org/10.1016/j.acha.2012.03.002
    $$R_{\delta}(A) = \| \{ \frac{\sigma_j}{\sigma_0} \ge \delta \}$$
    Where $\sigma_j$ denotes the $jth$ largest singular value of the matrix K
    TODO: This function currently assumes symmetricish distribution of eigenvalues.
    """
    eig, density = estimate_dos(A)
    approx_rank = np.maximum(
        np.max(density[np.where(-eig >= (thresh))]), 0
    ) + np.maximum(A.shape[0] - np.min(density[np.where(eig >= thresh)]), 0)

    return int(np.ceil(approx_rank))


def interpolative_decomposition(A, k, return_p=False):
    assert k < np.min(A.shape)
    q, r, perm = qr(A, pivoting=True)
    b = q[:, :k] @ r[:k, :k]
    if return_p:
        p = np.concatenate([np.eye(k), np.linalg.inv(r[:k, :k]) @ r[:k, k:]], axis=1)
        return b, p, perm
    return b


def approximate_rank_of_scales(A, thresh, scales):
    """ Returns one rank per scale, note that higher scales have less accuracy
    and may need more evaluations. Number of evaluations is currently set
    manually.
    """
    eig, density = estimate_dos(A)
    ranks = []
    for scale in scales:
        approx_rank = np.maximum(
            np.max(density[np.where(-eig >= (thresh ** (1 / scale)))]), 0
        ) + np.maximum(
            A.shape[0] - np.min(density[np.where(eig >= (thresh ** (1 / scale)))]), 0
        )
        approx_rank = int(np.ceil(approx_rank))
        ranks.append(approx_rank)
    return ranks


def apply_anisotropy(K, anisotropy):
    if anisotropy == 0:
        # do nothing
        return K

    if scipy.sparse.issparse(K):
        d = np.array(K.sum(1)).flatten()
        K = K.tocoo()
        K.data = K.data / ((d[K.row] * d[K.col]) ** anisotropy)
        K = K.tocsr()
        return K
    d = K.sum(1)
    K = K / (np.outer(d, d) ** anisotropy)
    return K


def apply_vectors(M, d, d_post=None):
    if d_post is None:
        d_post = d
    if scipy.sparse.issparse(M):
        M = M.tocoo()
        M.data = M.data * (d[M.row] * d_post[M.col])
        return M.tocsr()
    return M / np.outer(d, d_post)


def adjacency_to_operator(A, anisotropy):
    """ Gets the symmetric conjugate of the diffusion operator and its
    row/col sums as a vector.
    """
    M = apply_anisotropy(A, anisotropy)
    D_norm = np.array(M.sum(axis=0)).squeeze()
    return M, D_norm


def randomized_interpolative_decomposition(
    A, k_1, k_2, k_3=5, tol=1e-6, return_p=False
):
    """ Finds the columns of a large matrix that represent the whole matrix
    well in terms of rank. This is done by first projecting to k_2 (of order
    k_1) dimensions randomly, then doing QR decomposition. This results in a
    matrix S that (approximately) consists of a subset of size k_1 columns of
    W. To find the indices that S represents we then randomly project columns
    down to k_3 elements to quickly test for equality. This projection ensures
    the equality test is parallelizable and scales linearly with the size of W.
    Note that this equality test could fail for many reasons, including:
    (1) repeated columns in W, or columns that are within our tolerance of L_2
    distance.
    (2) k_3 is too small resulting in false positives (i.e. columns both in the
    null space of our projection that are not equal).
    """
    m, n = A.shape
    assert k_1 < k_2
    assert k_3 < k_2
    assert k_2 <= min(m, n)
    # if use_sparse:
    # J    sparse_interpolative_decomposition(A, k_1, return_p=return_p)
    # else:
    G = np.random.randn(k_2, A.shape[0])
    W = G @ A
    S = interpolative_decomposition(W, k_1, return_p=return_p)
    if return_p:
        S, P, perm = S
    indices = []
    R = np.random.randn(k_3, k_2)
    # Slow way, implemented more efficiently
    # Q = (R @ S)[:, :, None] - (R @ W)[:, None, :]
    count = 0
    while len(indices) != k_1:
        R = np.random.randn(k_3, k_2)
        # print(count, tol * (10**-count))
        indices = np.argwhere(
            np.linalg.norm((R @ S)[:, :, None] - (R @ W)[:, None, :], axis=0)
            < tol * (10 ** -count)
        )[:, 1]
        count += 1
        if count >= 10:
            indices = np.argwhere(
                np.linalg.norm(S[:, :, None] - W[:, None, :], axis=0) < tol
            )[:, 1]
            break
    assert len(indices) == k_1
    if return_p:
        return indices, P, perm
    return indices


class DiffusionEMD(object):
    """ Base class for DiffusionEMD estimators
    """

    def __init__(
        self,
        max_scale=10,
        n_scales=6,
        delta=0,
        anisotropy=1,
        alpha=0.5,
        min_basis=0,
        max_basis=None,
        **kwargs
    ):
        self.max_scale = max_scale
        # Filter does not tolerate scales below zero
        self.n_scales = min(n_scales, max_scale + 1)
        self.delta = delta
        self.anisotropy = anisotropy
        self.alpha = alpha
        self.min_basis = min_basis
        if max_basis is None:
            max_basis = np.inf
        self.max_basis = max_basis
        self.scales = [
            2 ** i for i in range(max_scale - self.n_scales + 1, max_scale + 1)
        ]
        assert 0 <= self.anisotropy <= 1

    def transform(self, y):
        pass

    def fit(self, X):
        self.X = X
        self.N = X.shape[0]
        self.M = apply_anisotropy(X, self.anisotropy)
        self.D = np.array(self.M.sum(axis=0)).squeeze()
        self.T = apply_vectors(self.M, self.D ** -0.5)

    def _compute_rank(self):
        self.basis_sizes = approximate_rank_of_scales(
            self.T, self.delta, scales=self.scales
        )
        self.basis_sizes = np.clip(self.basis_sizes, a_min=self.min_basis, a_max=None)

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, **kwargs)
        return self.transform(y)


class DiffusionTree(DiffusionEMD):
    def __init__(
        self,
        max_scale=10,
        n_scales=6,
        delta=0,
        anisotropy=1,
        alpha=0.5,
        min_basis=0,
        max_basis=None,
    ):
        n_scales = max_scale + 1
        super().__init__(
            max_scale=max_scale,
            n_scales=n_scales,
            delta=delta,
            anisotropy=anisotropy,
            alpha=alpha,
            min_basis=min_basis,
            max_basis=max_basis,
        )

    def fit(self, X):
        super().fit(X)
        self.T = apply_vectors(self.M, self.D ** -0.5)
        self._compute_rank()
        self._compute_diff_op()

    def _compute_diff_op(self):
        self.Ts = [self.T]
        self.Ps = [None]
        self.bases = [np.arange(self.N)]
        self.perms = [None]
        for j, arank in enumerate(self.basis_sizes[1:]):
            Tj = self.Ts[j]
            N = Tj.shape[0]
            # If arank is not significantly smaller, don't bother shrinking basis
            if arank < min(N * 0.5, self.max_basis):
                basis, P, perm = randomized_interpolative_decomposition(
                    Tj, arank, min(arank + 8, N), return_p=True
                )
                Tp1 = Tj[basis]
            else:
                P = None
                basis = np.arange(N)
                Tp1 = Tj
                perm = None
            self.perms.append(perm)
            self.Ts.append(Tp1 @ Tp1.transpose())
            self.Ps.append(P)
            self.bases.append(basis)

    def transform(self, y):
        dist_at_scale = y
        embeddings = []
        n_scales = len(self.scales)
        prev_diffusion = None
        for i, s in enumerate(self.scales):
            T = self.Ts[i]
            P = self.Ps[i]
            perm = self.perms[i]
            if P is not None:
                dist_at_scale = (
                    P
                    @ estimate_utils.permutation_vector_to_matrix(perm)
                    @ dist_at_scale
                )
            diffusion_at_scale = T @ dist_at_scale
            if P is not None:
                tmp = P.T @ diffusion_at_scale
            else:
                tmp = diffusion_at_scale
            if i > 0:
                weight = 0.5 ** ((n_scales - i) * self.alpha) * (
                    self.N / diffusion_at_scale.shape[0]
                )
                lvl_embed = weight * (tmp - prev_diffusion).T
                embeddings.append(lvl_embed)
            prev_diffusion = diffusion_at_scale
        embeddings.append(tmp.T)
        embeddings = np.concatenate(embeddings, axis=1)
        self.embeddings = embeddings
        return self.embeddings


class DiffusionCheb(DiffusionEMD):
    def __init__(
        self,
        max_scale=10,
        n_scales=6,
        delta=0,
        anisotropy=1,
        alpha=0.5,
        min_basis=0,
        max_basis=None,
        method="chebyshev",
        use_diff_wavelets=True,
        cheb_order=32,
    ):
        self.method = method
        self.use_diff_wavelets = use_diff_wavelets
        self.cheb_order = cheb_order
        super().__init__(
            max_scale=max_scale,
            n_scales=n_scales,
            delta=delta,
            anisotropy=anisotropy,
            alpha=alpha,
            min_basis=min_basis,
            max_basis=max_basis,
        )

    def fit(self, X):
        super().fit(X)
        graph = pygsp.graphs.Graph(self.M)
        # Use the normalized laplacian here for eigenvalues in [0, 2]
        graph.compute_laplacian("normalized")
        if self.method == "exact":
            graph.compute_fourier_basis()
        else:
            graph.estimate_lmax()

        kernels = [lambda x, s=s: np.minimum((1 - x) ** s, 1) for s in self.scales]
        self.filter = pygsp.filters.Filter(graph, kernels)

    def _subsample_embeddings(self, embeddings):
        # TODO make this work on a concatenated set of embeddings
        self.selections = [
            randomized_interpolative_decomposition(self.M, rank, min(rank + 8, self.N))
            if rank < self.max_basis
            else np.random.randint(self.N, size=rank)
            for rank in self.basis_sizes
        ]
        embeddings = [
            e[:, s] * self.M.shape[0] / a
            for s, e, a in zip(self.selections, embeddings, self.basis_sizes)
        ]
        return embeddings

    def transform(self, y):
        D_labels = (self.D[:, None] ** -0.5) * y
        diffusions = self.filter.filter(
            D_labels, method=self.method, order=self.cheb_order
        )
        diffusions = (self.D ** 0.5)[:, None, None] * diffusions
        n, n_samples, n_scales = diffusions.shape
        embeddings = []
        for k in range(n_scales):
            d = diffusions[..., k]
            if self.use_diff_wavelets:
                # Corresponds to Dual norm version (1) in Leeb and Coifman 2016
                if k < n_scales - 1:
                    d -= diffusions[..., k + 1]
                weight = 0.5 ** ((n_scales - k - 1) * self.alpha)
            else:
                # Corresponds to Dual norm version (2) in Leeb and Coifman 2016
                if k < n_scales - 1:
                    d -= diffusions[..., -1]
                weight = 0.5 ** ((n_scales - k - 1) * self.alpha)
            lvl_embed = weight * d.T
            embeddings.append(lvl_embed)
        if self.delta > 0:
            self._compute_rank()
            embeddings = self._subsample_embeddings(embeddings)
        else:
            self.basis_sizes = [n_samples] * n_scales
        self.embeddings = np.concatenate(embeddings, axis=1)
        return self.embeddings
