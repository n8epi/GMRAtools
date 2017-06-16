#!/usr/bin/env python
from distutils.core import setup
import os

# Utility function to read the README.md file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README.md file and 2) it's easier to type in the README.md file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "GMRAtools",
    version = "0.1",
    author = "Nate Strawn",
    author_email = "nate.strawn@georgetown.edu",
    description = ("Python library for Geometric Multiresolution Analysis."),
    license = "BSD",
    keywords = "geometric multiresolution analysis",
    url = "https://github.com/n8epi/GMRAtools",
    packages=['GMRAtools'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    dependency_links=['http://github.com/n8epi/CoverTree.git']
)
