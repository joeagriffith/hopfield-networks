import torch
from hopnet.activations import Activation

class Tanh(Activation):
    """
    This activation function is only used during training.
    It works as a replacement for the other activation functions
    when gradients need to be propagated through the network.
    Usage can be found in the training functions in hopnet.utils.train.py

    Args:
        coef (float): The coefficient to multiply the input by.
    """
    def __init__(self, coef=1.0):
        super(Tanh, self).__init__()
        self.coef = coef

    def __call__(self, x, _=None):
        """
        Applies the tanh function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return tanh(x, self.coef)

def tanh(x: torch.Tensor, coef=1.0):
    """
    A functional version of the tanh function.

    Args:
        x (torch.Tensor): The input tensor.
        coef (float): The coefficient to multiply the input by.

    Returns:
        torch.Tensor: The output tensor.
    """
    return torch.tanh(coef * x)
