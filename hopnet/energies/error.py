import torch
from hopnet.energies import Energy

"""
Inspired by Predictive Coding, the energy function allows for the problem to be reformulated as an energy minimization problem.
As the error is squared, the energy function is convex and is lower bounded by 0.
The energy is defined as the mean squared error between the current state and the next state of the network.
The network will be stable if it is in low energy.
"""
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