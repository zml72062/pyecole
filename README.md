# PyEcole - Python wrapper for Ecole

Ecole \[[github](https://github.com/ds4dm/ecole), [doc](https://doc.ecole.ai/py/en/stable/index.html)\], short for *Extensible Combinatorial Optimization Learning Environments*, is a collection of high-level interfaces whereby machine learning algorithms can work in collaboration with an [SCIP](https://www.scipopt.org/) solver throughout the solving procedure. 

PyEcole is merely a Python wrapper for Ecole that provides programmer-friendly features (e.g., method signatures, type hints, docstrings). Those features are absent in the original Ecole package, since Ecole has its bulk of code implemented in C++, and only provides Python interfaces through [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html). Thus, PyEcole can greatly ease programming with Ecole.

## Prerequisites

We use the following script to set up the environment.

```sh
conda create -n ml4co python==3.8
conda activate ml4co
conda install -c conda-forge ecole=0.7.3 scip=7.0.3 pyscipopt=3.3.0
```

Then run the following script to install `pyecole`.

```sh
git clone https://github.com/zml72062/pyecole
cd pyecole
python setup.py install
```
