"""
Handles datasets for the manifold OT project


"""
import graphtools
import numpy as np
from scipy.stats import special_ortho_group
import sklearn.datasets as skd
import sklearn.metrics
from sklearn.neighbors import kneighbors_graph
import ot
import pygsp


class Dataset(object):
    """ Dataset class for Optimal Transport

    Paramters
    ---------
    X: [N x F]

    labels: [N x M]

    """

    def __init__(self):
        super().__init__()
        self.X = None
        self.labels = None
        self.graph = None

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.X

    def standardize_data(self):
        """ Standardize data putting it in a unit box around the origin.
        This is necessary for quadtree type algorithms
        """
        X = self.X
        minx = np.min(self.X, axis=0)
        maxx = np.max(self.X, axis=0)
        self.std_X = (X - minx) / (maxx - minx)
        return self.std_X

    def rotate_to_dim(self, dim):
        """ Rotate dataset to a different dimensionality """
        self.rot_mat = special_ortho_group.rvs(dim)[: self.X.shape[1]]
        self.high_X = np.dot(self.X, self.rot_mat)
        return self.high_X

class Ring(Dataset):
    def __init__(self, n_points, random_state=42):
        super().__init__()
        self.n_points = n_points
        N = n_points
        self.random_state = random_state
        np.random.seed(42)
        self.X = np.linspace(0, 1 - (1 / N), N)[:, None]
        self.X_circle = np.stack([np.cos(2 * np.pi * self.X[:,0]), np.sin(2 * np.pi * self.X[:,0])], axis=1)
        # print(self.X_circle)
        #self.graph = pygsp.graphs.NNGraph(
        #    self.X_circle, epsilon=0.1, NNtype="radius", rescale=False, center=False
        #)
        self.graph = pygsp.graphs.Ring(self.n_points)
        self.labels = np.eye(N)

    def get_graph(self):
        return self.graph

class Line(Dataset):
    def __init__(self, n_points, random_state=42):
        super().__init__()
        self.n_points = n_points
        N = n_points
        self.random_state = random_state
        np.random.seed(42)
        self.X = np.linspace(0, 1, N)[:, None]
        # self.X_circle = np.stack(
        #     [np.cos(2 * np.pi * self.X[:, 0]), np.sin(2 * np.pi * self.X[:, 0])],
        #     axis=1
        # )
        self.graph = pygsp.graphs.NNGraph(
            self.X, epsilon=0.1, NNtype="radius", rescale=False, center=False
        )
        self.labels = np.eye(N)

    def get_graph(self):
        return self.graph


class SklearnDataset(Dataset):
    """ Make a dataset based on an SKLearn dataset with a
    gaussian centered at each point.
    """

    def __init__(
        self,
        name=None,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.0,
        random_state=42,
    ):
        super().__init__()
        self.name = name
        self.n_distributions = n_distributions
        self.n_points_per_distribution = 50
        self.noise = noise
        self.random_state = random_state
        if name == "swiss_roll":
            f = skd.make_swiss_roll
        elif name == "s_curve":
            f = skd.make_s_curve
        else:
            raise NotImplementedError("Unknown sklearn dataset: %s" % name)
        self.means, self.t = f(
            n_samples=n_distributions, noise=noise, random_state=random_state
        )
        rng = np.random.default_rng(random_state)

        clouds = np.array(
            [
                rng.multivariate_normal(
                    mean, 20 * np.identity(3), n_points_per_distribution
                )
                for mean in self.means
            ]
        )
        self.X = np.reshape(clouds, (n_distributions * n_points_per_distribution, 3))
        self.labels = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )

    def get_graph(self):
        """ Create a graphtools graph if does not exist
        """
        if self.graph is None:
            self.graph = graphtools.Graph(self.X, use_pygsp=True)
        return self.graph

class SwissRoll(Dataset):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.0,
        manifold_noise=1.0,
        width=1,
        random_state=42,
    ):
        super().__init__()
        rng = np.random.default_rng(random_state)

        mean_t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=(1, n_distributions)))
        mean_y = width * rng.uniform(size=(1, n_distributions))
        t_noise = (
            manifold_noise
            * 3
            * rng.normal(size=(n_distributions, n_points_per_distribution))
        )
        y_noise = (
            manifold_noise
            * 7
            * rng.normal(size=(n_distributions, n_points_per_distribution))
        )
        ts = np.reshape(t_noise + mean_t.T, -1)
        ys = np.reshape(y_noise + mean_y.T, -1)
        xs = ts * np.cos(ts)
        zs = ts * np.sin(ts)
        X = np.stack((xs, ys, zs))
        X += noise * rng.normal(size=(3, n_distributions * n_points_per_distribution))
        self.X = X.T
        self.ts = np.squeeze(ts)
        self.labels = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        self.t = mean_t[0]
        mean_x = mean_t * np.cos(mean_t)
        mean_z = mean_t * np.sin(mean_t)
        self.means = np.concatenate((mean_x, mean_y, mean_z)).T

    def get_graph(self):
        """ Create a graphtools graph if does not exist
        """
        if self.graph is None:
            self.graph = graphtools.Graph(self.X, use_pygsp=True)
        return self.graph


class Sphere(Dataset):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        dim = 3,
        noise=0.05,
        label_noise = 0.0,
        manifold_noise=1.0,
        width=1,
        flip=False,
        random_state=42,
    ):
        super().__init__()
        self.n_distributions = n_distributions
        self.n_points_per_distribution = n_points_per_distribution
        self.dim = dim
        self.noise = noise
        self.manifold_noise = manifold_noise
        rng = np.random.default_rng(random_state)

        X = rng.normal(0, 1, (self.dim, self.n_distributions))
        X = X / np.linalg.norm(X, axis=0)
        self.means = X.T
        X = X[:, :, None]
        X = np.repeat(X, n_points_per_distribution, axis=-1)
        noise = noise * rng.normal(size = (dim, n_distributions, n_points_per_distribution))
        X += noise
        X = X.reshape(dim, -1)
        X = X / np.linalg.norm(X, axis=0)
        #X += noise * rng.normal(size=(self.dim, n_distributions, n_points_per_distribution))

        self.X = X.T
        self.labels = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        
        # Flipping noise
        if flip:
            index_to_flip = np.random.randint(n_distributions * n_points_per_distribution, size = n_distributions)
            for i in range(n_distributions):
                self.labels[index_to_flip[i], i] = 1 - self.labels[index_to_flip[i], i]
            self.labels = self.labels / np.sum(self.labels, axis=0)


        # Ground truth dists (approximate) and clip for numerical errors
        self.gtdists = np.arccos(np.clip(self.means @ self.means.T, 0, 1))

    def get_graph(self):
        """ Create a graphtools graph if does not exist
        """
        if self.graph is None:
            #self.graph = graphtools.Graph(self.X, use_pygsp=True, knn=10)
            self.graph = pygsp.graphs.NNGraph(
                self.X, epsilon=0.1, NNtype="radius", rescale=False, center=False
            )
            #self.graph = graphtools.Graph(self.X, use_pygsp=True, knn=100)
        return self.graph

class Mnist(Dataset):
    def __init__(self):
        from torchvision.datasets import MNIST
        self.mnist_train = MNIST("/home/atong/data/mnist/", download=True)
        self.mnist_test = MNIST("/home/atong/data/mnist/", download=True, train=False)
        self.graph = pygsp.graphs.Grid2d(28, 28)

    def get_graph(self):
        return self.graph


