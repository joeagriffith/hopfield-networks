import torch
from hopnet.activations import Activation

"""
This activation function is only used during training.
It works as a replacement for the other activation functions
when gradients need to be propagated through the network.
Usage can be found in the training functions in hopnet.utils.train.py
"""
class Tanh(Activation):
    def __init__(self, mult=1.0):
        super(Tanh, self).__init__()
        self.mult = mult

    def __call__(self, x, _=None):
        return tanh(x, self.mult)

def tanh(x: torch.Tensor, mult=1.0):
    return torch.tanh(mult * x)
