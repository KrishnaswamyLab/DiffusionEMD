from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import graphtools
import sklearn.datasets
import pygsp
import sklearn
import ot
import pandas as pd
import scipy.sparse
import pickle
import scprep
from manifold_ot import estimate_utils


class Spherical_MNIST_Predictions:
    def __init__(self, sphere_graph_path, sphere_signals_path):
        if sphere_graph_path and sphere_signals_path:
            self.sphere_graph = pickle.load(sphere_graph_path)
            self.sphere_signals = pickle.load(sphere_signals_path)
        else:
            print("Please precompute the sphere -- not yet implemented")
        self.knn = KNeighborsClassifier(n_neighbors=1)

    def knn_classify(self, embeddings, num_neighbors=1):
        # perform a train/test split (by default 50-50)
        knn = KNeighborsClassifier(n_neighbors=num_neighbors)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, self.dataset_labels, random_state=0
        )
        knn.fit(X_train, y_train)
        # get prediction accuracy
        preds = knn.predict(X_test)
        acc = np.sum((preds == y_test).float()) / len(X_test)
        return acc

    def MOT_embedding(self, num_evals=1000):
        # perform a MOT embedding of the dataset
        def apply_anisotropy(K, anisotropy):
            if anisotropy == 0:
                # do nothing
                return K
            if scipy.sparse.issparse(K):
                d = np.array(K.sum(1)).flatten()
                K = K.tocoo()
                K.data = K.data / ((d[K.row] * d[K.col]) ** anisotropy)
                K = K.tocsr()
                return K, d
            d = K.sum(1)
            K = K / (np.outer(d, d) ** anisotropy)
            return K, d

        def apply_vectors(M, d, d_post=None):
            if d_post is None:
                d_post = d
            if scipy.sparse.issparse(M):
                M = M.tocoo()
                M.data = M.data * (d[M.row] * d_post[M.col])
                return M.tocsr()
            return M / np.outer(d, d_post)

        def diffusion_embeddings(
            graph,
            distribution_labels,
            method="chebyshev",
            max_scale=7,
            min_scale=1,
            version=1,
            anisotropy=0.0,
            k=None,
            return_eig=False,
            subselect=False,
            alpha=1,
        ):
            """
            Return the vectors whose L1 distances are the EMD between the given distributions.
            The graph supplied (a PyGSP graph) should encompass both distributions.
            The distributions themselves should be one-hot encoded with the distribution_labels parameter.
            """
            assert version >= 3
            assert 0 <= anisotropy <= 1
            if k is None:
                k = graph.N - 1
                print(f"Graph has N = {graph.N}. Using k = {k}")
            diffusions = []
            if version <= 4:
                graph.compute_laplacian(lap_type="normalized")
                # Lazy symmetric random walk matrix
                P = np.eye(graph.N) - graph.L / 2
                # e, U = np.linalg.eigh(P)
                e, U = scipy.sparse.linalg.eigsh(P, k=k)
                for scale in [2 ** i for i in range(1, max_scale)]:
                    Pt = U @ np.diag(e ** scale) @ U.T
                    diffusions.append(Pt @ distribution_labels)
            else:
                A = graph.W
                D = np.array(A.sum(axis=0)).squeeze()
                P = apply_anisotropy(A, anisotropy)
                # Sums along axis=1 are all 1
                D_norm = np.array(P.sum(axis=0)).squeeze()
                M = apply_vectors(P, D_norm ** -0.5)
                e, U = scipy.sparse.linalg.eigsh(M, k=k)
                for scale in [2 ** i for i in range(min_scale, max_scale)]:
                    Pt_sym = U @ np.diag(e ** scale) @ U.T
                    Pt = apply_vectors(Pt_sym, D_norm ** -0.5, D_norm ** 0.5)
                    diffusions.append(Pt @ distribution_labels)
            diffusions = np.stack(diffusions, axis=-1)
            n, n_samples, n_scales = diffusions.shape
            embeddings = []
            for i in range(n_scales):
                d = diffusions[..., i]
                if (version == 2) or (version == 3):
                    if i < n_scales - 1:
                        d -= diffusions[..., -1]
                    weight = 0.5 ** (n_scales - i - 1)
                elif version == 4:
                    if i < n_scales - 1:
                        d -= diffusions[..., i + 1]
                    weight = 0.5 ** (n_scales - i - 1)
                elif version == 5:
                    if i < n_scales - 1:
                        d -= diffusions[..., -1]
                    weight = 0.5 ** ((n_scales - i - 1) * alpha)
                elif version == 6:
                    if i < n_scales - 1:
                        d -= diffusions[..., i + 1]
                    weight = 0.5 ** ((n_scales - i - 1) * alpha)
                lvl_embed = weight * d.T

                embeddings.append(lvl_embed)

            if subselect:
                num_samples = approximate_rank_of_scales(
                    P, 0.5, scales=[2 ** i for i in range(min_scale, max_scale)]
                )
                print(num_samples)
                augmented_num_samples = [
                    min(n * (2 ** (i + min_scale)), graph.N)
                    for i, n in enumerate(num_samples)
                ]
                print(augmented_num_samples)
                selections = []
                pps = []
                for arank in augmented_num_samples:
                    selected, pp = randomized_interpolative_decomposition(
                        np.array(P), arank, arank + 8, return_p=True
                    )
                    selections.append(selected)
                    pps.append(pp)
                # augmented_num_samples
                print(embeddings[0].shape, len(embeddings))
                tmp = []
                for s, e, a in zip(selections, embeddings, augmented_num_samples):
                    tmp.append(e[:, s] * graph.N / a)
                embeddings = tmp
            embeddings = np.concatenate(embeddings, axis=1)
            if return_eig and subselect:
                return embeddings, e, U, pps
            if return_eig:
                return embeddings, e, U
            return embeddings

        embeddings = diffusion_embeddings(
            self.sphere_graph, self.sphere_signals, version=5, max_scale=12, k=num_evals
        )
        return embeddings
