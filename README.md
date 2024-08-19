# PyEcole - Python wrapper for Ecole

Ecole \[[github](https://github.com/ds4dm/ecole), [doc](https://doc.ecole.ai/py/en/stable/index.html)\], short for *Extensible Combinatorial Optimization Learning Environments*, is a collection of high-level interfaces whereby machine learning algorithms can work in collaboration with an [SCIP](https://www.scipopt.org/) solver throughout the solving procedure. 

PyEcole is merely a Python wrapper for Ecole that provides programmer-friendly features (e.g., method signatures, type hints, docstrings). Those features are absent in the original Ecole package, since Ecole has its bulk of code implemented in C++, and only provides Python interfaces through [`pybind11`](https://pybind11.readthedocs.io/en/stable/index.html). Thus, PyEcole can greatly ease programming with Ecole.

## Getting started

For environment setup, please refer to the [contribution guidelines](https://doc.ecole.ai/py/en/stable/contributing.html) of Ecole. First, clone the github repository of Ecole: (a fork of the official release)
```sh
git clone https://github.com/zml72062/ecole
```
At the root directory of the repository, run the following commands to tidy up dependencies for Ecole:
```sh
conda env create -n ecole -f dev/conda.yaml
conda activate ecole
conda config --append channels conda-forge
conda config --set channel_priority flexible
```
Then install Ecole by running
```sh
python -m pip install --no-deps --no-build-isolation .
```
After everything above is done, run the following script to install `pyecole`:
```sh
git clone https://github.com/zml72062/pyecole
cd pyecole
python setup.py install
```
