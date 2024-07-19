import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyecole",
    version = "0.0.1",
    author = "zml72062",
    author_email = "zml72062@126.com",
    description = ("Python wrapper for Ecole."),
    url = "https://github.com/zml72062/pyecole",
    packages=find_packages(include=["pyecole", "pyecole.*"]),
    long_description=read('README.md'),
)
