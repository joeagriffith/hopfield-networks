import torch
from hopnet.activations import Activation

class Tanh(Activation):
    def __init__(self, mult=1.0):
        super(Tanh, self).__init__()
        self.mult = mult

    def __call__(self, x, _=None):
        return tanh(x, self.mult)

def tanh(x: torch.Tensor, mult=1.0):
    return torch.tanh(mult * x)
