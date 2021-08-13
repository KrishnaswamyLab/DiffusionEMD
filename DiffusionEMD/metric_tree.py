""" metric_tree.py
This file uses sklearn trees generally used for KNN calculation as an
approximate metric tree for wasserstein distance.  Further extensions are
quadtree, and one based on hierarchical clustering.  The idea is to use the
tree with edge lengths as the (L2) distance between means.  The distance
between any two points embedded in this tree is then the geodesic distance
along the tree.  Note that this is an offline algorithm, we do not support
adding points after the initial construction.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.neighbors import KDTree, BallTree, DistanceMetric
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import coo_matrix


class QuadTree(object):
    """
    This quadtree could be sped up, but is an easy implementation
    """

    def __init__(self, X, n_levels=25, noise=1.0, *args, **kwargs):
        assert np.all(np.min(X, axis=0) >= 0)
        assert np.all(np.max(X, axis=0) <= 1)
        assert n_levels >= 1
        self.kwargs = kwargs
        self.X = X
        self.noise = noise
        # self.X = self.X + np.random.randn(*self.X.shape) * noise
        self.dims = X.shape[1]
        self.n_clusters = 2 ** self.dims
        self.n_levels = n_levels
        center = np.random.rand(self.dims) * noise
        self.tree, self.indices, self.centers, self.dists = self._cluster(
            center, np.arange(X.shape[0]), n_levels=self.n_levels - 1, start=0
        )
        self.tree = [(0, self.X.shape[0], n_levels, 0), *self.tree]
        self.dists = np.array([0, *self.dists])
        self.centers = [center, *self.centers]
        self.centers = np.array(self.centers)

    def _cluster(self, center, index, n_levels, start):
        """
        Parameters
        ----------

        bounds:
            [2 x D] matrix giving min / max of bounding box for this cluster

        """
        if n_levels == 0 or len(index) == 0:
            return None
        labels = np.ones_like(index) * -1
        dim_masks = np.array([self.X[index, d] > center[d] for d in range(self.dims)])
        import itertools

        bin_masks = np.array(list(itertools.product([False, True], repeat=self.dims)))
        label_masks = np.all(bin_masks[..., None] == dim_masks[None, ...], axis=1)
        for i, mask in enumerate(label_masks):
            labels[mask] = i
        assert np.all(labels > -1)
        shift = 2 ** -(self.n_levels - n_levels + 2)
        shifts = np.array(list(itertools.product([-shift, shift], repeat=self.dims)))
        cluster_centers = shifts + center
        sorted_index = []
        children = []
        ccenters = []
        cdists = []
        is_leaf = [0] * self.n_clusters
        unique, ucounts = np.unique(labels, return_counts=True)
        counts = np.zeros(self.n_clusters, dtype=np.int32)
        for u, c in zip(unique, ucounts):
            counts[u] = c
        cstart = 0
        for i, count, ccenter in zip(unique, counts, cluster_centers):
            ret = self._cluster(
                ccenter, index[labels == i], n_levels - 1, start + cstart
            )
            if ret is None:
                sorted_index.extend(index[labels == i])
                is_leaf[i] = 1
                continue
            sorted_index.extend(ret[1])
            children.extend(ret[0])
            ccenters.extend(ret[2])
            cdists.extend(ret[3])
            cstart += count

        to_return = list(
            zip(
                *[
                    np.array([0, *np.cumsum(counts)]) + start,
                    np.cumsum(counts) + start,
                    [n_levels] * self.n_clusters,
                    is_leaf,
                ]
            )
        )
        dists = np.linalg.norm(cluster_centers - center[None, :], axis=1)
        return (
            [*to_return, *children],
            sorted_index,
            [*cluster_centers, *ccenters],
            [*dists, *cdists],
        )

    def get_arrays(self):
        return None, self.indices, self.tree, self.centers, self.dists


class ClusterTree(object):
    def __init__(self, X, n_clusters=10, n_levels=5, *args, **kwargs):
        self.X = X
        self.n_clusters = n_clusters
        self.n_levels = n_levels
        center = self.X.mean(axis=0)
        self.tree, self.indices, self.centers, self.dists = self._cluster(
            center, np.arange(X.shape[0]), n_levels=self.n_levels - 1, start=0
        )
        self.tree = [(0, self.X.shape[0], n_levels, n_levels == 1), *self.tree]
        self.centers = [center, *self.centers]
        self.dists = np.array([0, *self.dists])
        self.centers = np.array(self.centers)

    def _cluster(self, center, index, n_levels, start):
        """
        Returns a list of tuples corresponding to each subnode of the tree
        (center, level, start, end, is_leaf), sorted_index
        center is the cluster center
        level is the level of the node counting the root as the zeroth level
        sorted_index is athe list of
        """
        if n_levels == 0 or len(index) < self.n_clusters:
            return None
        cl = MiniBatchKMeans(n_clusters=self.n_clusters)
        cl.fit(self.X[index])
        sorted_index = []
        children = []
        ccenters = []
        cdists = []
        is_leaf = [0] * self.n_clusters
        unique, ucounts = np.unique(cl.labels_, return_counts=True)
        counts = np.zeros(self.n_clusters, dtype=np.int32)
        for u, c in zip(unique, ucounts):
            counts[u] = c
        cstart = 0
        for i, count in zip(unique, counts):
            ret = self._cluster(
                cl.cluster_centers_[i],
                index[cl.labels_ == i],
                n_levels - 1,
                start + cstart,
            )
            if ret is None:
                sorted_index.extend(index[cl.labels_ == i])
                is_leaf[i] = 1
                continue
            sorted_index.extend(ret[1])
            children.extend(ret[0])
            ccenters.extend(ret[2])
            cdists.extend(ret[3])
            cstart += count
        to_return = list(
            zip(
                *[
                    np.array([0, *np.cumsum(counts)]) + start,
                    np.cumsum(counts) + start,
                    [n_levels] * self.n_clusters,
                    is_leaf,
                ]
            )
        )
        dists = np.linalg.norm(cl.cluster_centers_ - center[None, :], axis=1)
        return (
            [*to_return, *children],
            sorted_index,
            [*cl.cluster_centers_, *ccenters],
            [*dists, *cdists],
        )

    def get_arrays(self):
        return None, self.indices, self.tree, self.centers, self.dists


class MetricTree(BaseEstimator):
    def __init__(self, tree_type="ball", leaf_size=40, metric="euclidean", **kwargs):
        self.tree_type = tree_type
        if tree_type == "ball":
            self.tree_cls = BallTree
        elif tree_type == "kd":
            self.tree_cls = KDTree
        elif tree_type == "cluster":
            self.tree_cls = ClusterTree
        elif tree_type == "quad":
            self.tree_cls = QuadTree
        else:
            raise NotImplementedError("Unknown tree type")
        self.kwargs = kwargs
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_fn = DistanceMetric.get_metric(metric)

    def get_node_weights(self):
        """ Takes the middle of the bounds as the node center for each node
        TODO (alex): This could be improved or at least experimented with
        """
        node_weights = self.tree.get_arrays()[-1]
        if self.tree_type == "ball":
            return node_weights[0]
        elif self.tree_type == "kd":
            return node_weights.mean(axis=0)
        elif self.tree_type == "cluster":
            return node_weights
        elif self.tree_type == "quad":
            return node_weights
        else:
            raise NotImplementedError("Unknown tree type")

    def fit_transform(self, X, y):
        """
        X is data array (np array)
        y is one-hot encoded distribution index (np array of size # points x #
        distributions.
        """
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        self.classes_ = y.shape[1]  # unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.tree = self.tree_cls(
            X, leaf_size=self.leaf_size, metric=self.metric, **self.kwargs
        )

        tree_indices = self.tree.get_arrays()[1]
        node_data = self.tree.get_arrays()[2]
        y_indices = y[tree_indices]  # reorders point labels by tree order.

        self.edge_weights = self.get_node_weights()
        counts = np.empty((len(node_data), y.shape[1]))
        for node_idx in reversed(range(len(node_data))):
            start, end, is_leaf, radius = node_data[node_idx]

            # # Get counts represented in a sparse way through counters
            # counts[node_idx] = Counter(y_indices[start:end])

            # Find the number of points present in this range from each distribution
            counts[node_idx] = np.sum(
                y_indices[start:end], axis=0
            )  # as y is a one-hot encoding, we just need to sum over the relevant bits.

            # if is_leaf:
            #    counts[node_idx] = Counter(y_indices[start:end])
            # else:
            #    counts[node_idx] = counts[node_idx * 2 + 1] + counts[node_idx * 2 + 2]

            # Get edge weight
            """
            if node_idx > 0:
                edge_weights[node_idx] = self.dist_fn.pairwise(
                    [self.node_centers[node_idx]],
                    [self.node_centers[(node_idx - 1) // 2]],
                )
            """

        # convert to COO format
        dim = (self.classes_, len(node_data))
        dist_list = np.arange(1, self.classes_ + 1)
        self.counts_mtx = coo_matrix(dim, dtype=np.int32)
        for i, count in enumerate(counts):
            if np.sum(count) == 0:  # if no classes have signals in this region
                continue
            # get the signals with nonzero representation in the region
            # count is a list of the representation per distribution.
            # count_copy is used to eliminate distributions without representation
            count_copy = count.copy()
            count_copy[count_copy > 0] = 1
            dists_represented = np.multiply(dist_list, count_copy)
            j_list = (
                dists_represented[dists_represented != 0] - 1
            )  # we added 1 to the distribution numbers to do the zero trick.
            val_list = count[count != 0]
            i_list = [i] * len(j_list)
            self.counts_mtx += coo_matrix(
                (val_list, (j_list, i_list)), shape=dim, dtype=np.int32
            )

        return self.counts_mtx, self.edge_weights

    def transform(self, X):
        """ Transforms datasets y to (L1) vector space.

        Returns vectors representing edge weights and weights over vector.
        """
        check_is_fitted(self, "X_")

        if X != self.X_:
            raise ValueError("X transformed must equal fitted X")


if __name__ == "__main__":
    mt = MetricTree(tree_type="cluster")
    gt = np.repeat(np.arange(10), 100)
    gt = (
        (np.repeat(np.arange(max(gt) + 1)[:, None], len(gt), axis=1) == gt)
        .astype(int)
        .T
    )
    counts, edge_weights = mt.fit_transform(X=np.random.random_sample((1000, 3)), y=gt)
    print(counts, edge_weights)
    print(counts.toarray()[:50])
