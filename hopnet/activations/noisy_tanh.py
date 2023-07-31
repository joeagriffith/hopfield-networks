import torch
from hopnet.activations import Activation

class NoisyTanh(Activation):
    """
    This activation function is only used during training.
    It works as a replacement for the other activation functions
    when gradients need to be propagated through the network.
    Usage can be found in the training functions in hopnet.utils.train.py

    Args:
        coef (float): The coefficient to multiply the input by.
    """
    def __init__(self, coef=1.0, temperature=1.0):
        super(NoisyTanh, self).__init__()
        self.coef = coef
        self.temperature = temperature

    def __call__(self, x, step_i=int):
        """
        Applies the tanh function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return noisy_tanh(x, step_i, self.coef)

def noisy_tanh(x: torch.Tensor, step_i, coef=1.0,):
    """
    A functional version of the tanh function.

    Args:
        x (torch.Tensor): The input tensor.
        coef (float): The coefficient to multiply the input by.

    Returns:
        torch.Tensor: The output tensor.
    """

    noise = (torch.randn(x.shape) * (0.1 / (2.0 * step_i + 1.0))).to(x.device)

    return torch.tanh(coef * x + noise)
