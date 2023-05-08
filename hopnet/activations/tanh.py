import torch
from hopnet.activations import Activation

class Tanh(Activation):
    def __call__(self, x, _=None):
        return tanh(x)

def tanh(x: torch.Tensor):
    return torch.tanh(x)
