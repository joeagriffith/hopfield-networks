import torch
from hopnet.energies import Energy

class ErrorEnergy(Energy):
    """
    Inspired by Predictive Coding, the energy function allows for the problem to be reformulated as an energy minimization problem.
    As the error is squared, the energy function is convex and is lower bounded by 0.
    The energy is defined as the mean squared error between the current state and the next state of the network.
    The network will be stable if it is in low energy.

    Args:
        actv_fn (Activation): The activation function to apply to the next state of the network.
    """
    def __init__(self, actv_fn=None):
        self.actv_fn = actv_fn

    def __call__(self, x, weight, bias=None):
        """
        Calculates the energy of the network in its current state.
        This energy is the mean squared error between the current state 
        and the next state of the network if it were to be updated following the standard Hopfield update rule.

        Args:
            x (torch.Tensor): The current state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
            weight (torch.Tensor): The weight matrix of the network. Must be a 2D tensor (N, N).
            bias (torch.Tensor): The bias vector of the network. If None, no bias is applied. If not None, must be a 1D tensor (N).

        Returns:
            torch.Tensor: The energy of the network in its current state. A 1D tensor of length batch_size.
        """
        return error_energy(x, weight, bias, self.actv_fn)

def error_energy(x, weight, bias=None, actv_fn=None):
    """
    The error energy function.

    Args:
        x (torch.Tensor): The current state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
        weight (torch.Tensor): The weight matrix of the network. Must be a 2D tensor (N, N).
        bias (torch.Tensor): The bias vector of the network. If None, no bias is applied. If not None, must be a 1D tensor (N).
        actv_fn (Activation): The activation function to apply to the next state of the network.

    Returns:
        torch.Tensor: The energy of the network in its current state. A 1D tensor of length batch_size.
    """
    next_x = x @ weight
    if bias is not None:
        next_x += bias
    if actv_fn is not None:
        next_x = actv_fn(next_x)

    return (x - next_x).square().mean(dim=1)