import torch
from hopnet.activations import Activation

class Identity(Activation):
    """
    A placeholder activation function that does nothing.
    """
    def __call__(self, x, _=None):
        """
        Applies the identity function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            _ (None): Ignored.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        return identity(x)

def identity(x: torch.Tensor):
    """
    A functional version of the identity function.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.kk
    """
    return x