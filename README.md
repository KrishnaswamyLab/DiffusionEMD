# Implementation of Diffusion EMD

## Installation

To install from github use the following:

```
pip install git+https://github.com/KrishnaswamyLab/DiffusionEMD
```

Requirements can be found in `requirements.txt`

## Examples

Examples are in the `notebooks` directory.

Take a look at the examples provided there to get a sense of how the parameters
behave on simple examples that are easy to visualize.

## Usage

DiffusionEMD is written following the `sklearn` estimator framework. We provide two functions that operate quite differently. First the Chebyshev approxiamtion of the operator in `DiffusionCheb`, which we recommend when the number of distributions is small compared to the number of points. Second, the Interpolative Decomposition method that computes dyadic powers of $P^{2^k}$ directly in `DiffusionTree`. These two classes are used in the same way, first supplying parameters, fitting to a graph and array of distributions:

```
from DiffusionEMD import DiffusionCheb

dc = DiffusionCheb()
embeddings = dc.fit_transform(Adjacency, Distributions)

```

Where `Adjacency` is a $n \times n$ sparse `scipy` matrix, and `Distributions` is a $m \times n$ matrix of $m$ distributions over the $n$ graph nodes.
