# Hopfield Networks

## Getting started

An example script is supplied, 'mnist_discrete.ipynb' which includes detailed comments for how it may be modified.
It shows how the package may be used to remember and reconstruct binary images from the mnist dataset.

## Description
This repository is a modular package for implmenting Hopfield Networks.
It allows for different activation functions, energy functions, network architectures and training methods.
The package is written in Python and relies mostly on PyTorch for its computation.

## Installation
The 'hopnet' package only depends on PyTorch and tqdm. 
Pytorch should be installed by following the information on their website at: https://pytorch.org/get-started/locally/
Tqdm can be installed simply as 'pip install tqdm'.
If the aforementioned dependencies are installed, you should be able to clone the repository and run the 'mnist_discrete' notebook yourself.


## Usage
When building a model, the user must determine the: model, activation function, energy function and training mode.
How this is done is best exemplified in the 'mnist_discrete' notebook. I encourage a deeper exploration into the code
which is well structured and heavily commented. Attention should be given to the activations/, energies/ and models/ directories, aswell as Utils/train.py.
The other files in Utils are mostly helper functions and evaluation functions.