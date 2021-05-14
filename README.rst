Implementation of Diffusion EMD
===============================

Diffusion Earth Mover's Distance embeds the Wasserstein distance between two distributions on a graph into :math:`L^1` in log-linear time.

Installation
------------

DiffusionEMD is available in `pypi`. Install by running the following::

    pip install DiffusionEMD

Quick Start
-----------


DiffusionEMD is written following the `sklearn` estimator framework. We provide two functions that operate quite differently. First the Chebyshev approxiamtion of the operator in `DiffusionCheb`, which we recommend when the number of distributions is small compared to the number of points. Second, the Interpolative Decomposition method that computes dyadic powers of $P^{2^k}$ directly in `DiffusionTree`. These two classes are used in the same way, first supplying parameters, fitting to a graph and array of distributions::

    import numpy as np
    from DiffusionEMD import DiffusionCheb

    # Setup an adjacency matrix and a set of distributions to embed
    adj = np.ones((10, 10))
    distributions = np.random.randn(10, 5)
    dc = DiffusionCheb()

    # Embeddings where the L1 distance approximates the Earth Mover's Distance
    embeddings = dc.fit_transform(adj, distributions)
    # Shape: (5, 60)

Requirements can be found in `requirements.txt`

Examples
--------

Examples are in the `notebooks` directory.

Take a look at the examples provided there to get a sense of how the parameters
behave on simple examples that are easy to visualize.

Paper
-----

This code implements the algorithms described in this paper:

ArXiv Link: http://arxiv.org/abs/2102.12833::

    @inproceedings{tong_diffusion_2021,
      title = {Diffusion {{Earth Mover}}'s {{Distance}} and {{Distribution Embeddings}}},
      author = {Tong, Alexander and Huguet, Guillaume and Natik, Amine and MacDonald, Kincaid and Kuchroo, Manik and Coifman, Ronald and Wolf, Guy and Krishnaswamy, Smita},
      year = {2021},
      archiveprefix = {arXiv},
      eprint = {2102.12833},
    }
