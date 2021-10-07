from setuptools import find_packages, setup

import os

install_requires = [
    "numpy>=1.16.0",
    "scipy",
    "matplotlib>=3.0",
    "pot",
    "pygsp",
    "graphtools",
]

doc_requires = [
    "sphinx",
    "sphinxcontrib-napoleon",
    "ipykernel",
    "nbsphinx",
    "autodocsumm",
]

test_requires = [
    "nose",
    "nose2",
    "coverage",
    "coveralls",
    "parameterized",
    "requests",
    "packaging",
    "mock",
    "matplotlib>=3.0",
    "black",
]

version_py = os.path.join(os.path.dirname(__file__), "DiffusionEMD", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="DiffusionEMD",
    packages=find_packages(),
    version=version,
    description="Diffusion based earth mover's distance.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "doc": doc_requires,
    },
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/DiffusionEMD",
)
