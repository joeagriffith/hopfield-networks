# Hopfield Networks

## Getting started


## Description
This repository is a modular package for implementing Hopfield Networks.

It allows for different activation functions, energy functions, network architectures and training methods.

The package is written in Python and relies mostly on PyTorch for its computation.

## Installation
The 'hopnet' package only depends on PyTorch, Torchvision, tkinter and tqdm. 

Pytorch and Torchvision should be installed by following the information on their website at: https://pytorch.org/get-started/locally/

Tqdm can be installed simply as 'pip install tqdm'.

tkinter can be installed using 'pip install tkinter'.

If the aforementioned dependencies are installed, you should be able to clone the repository and run the 'mnist_discrete' notebook yourself.

## Documentation
The package has fully complete HTML documentation which can be accessed at hopnet/docs/_build by opening any of the html files.

I apologise if its not the most standard, but its my first time documenting code this way. 

There's a search bar so the user can search for any function/class and see the relevant documentation on it.

## Usage
When building a model, the user must determine the: model, activation function, energy function and training mode.

How this is done is best exemplified in the 'mnist_discrete' notebook. It includes detailed comments for how it may be modified

I encourage a deeper exploration into the code which is well structured and heavily commented. 

Attention should be given to the activations/, energies/ and models/ directories, aswell as Utils/train.py.

The other files in Utils are mostly helper functions and evaluation functions.