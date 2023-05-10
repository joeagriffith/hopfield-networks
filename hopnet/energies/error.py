import torch
from hopnet.energies import Energy

class ErrorEnergy(Energy):
    def __init__(self, actv_fn=None):
        self.actv_fn = actv_fn

    def __call__(self, x, weight, bias=None):
        return error_energy(x, weight, bias, self.actv_fn)

def error_energy(x, weight, bias=None, actv_fn=None):
    next_x = x @ weight
    if bias is not None:
        next_x += bias
    if actv_fn is not None:
        next_x = actv_fn(next_x)

    return (x - next_x).square().mean(dim=1)