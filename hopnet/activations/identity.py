import torch
from hopnet.activations import Activation

class Identity(Activation):
    def __call__(self, x, _=None):
        return identity(x)

def identity(x: torch.Tensor):
    return x