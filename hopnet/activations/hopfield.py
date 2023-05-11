import torch
from hopnet.activations import Activation
from hopnet.utils.functional import binary_to_spin

class HopfieldActivation(Activation):
    """
    The standard Hopfield activation function. 
    Essentially function like the sign function, 
    yet the user must specify a prefered sign for when the input is 0.

    Args:
        prefer (int): The prefered sign for when the input is 0.

    Raises:
        AssertionError: If prefer is not -1 or 1
    """
    def __init__(self, prefer:int):
        assert prefer in [-1, 1], "prefer must be -1 or 1"
        self.prefer = prefer

    def __call__(self, x, _=None):
        """
        Applies the Hopfield activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            _ (None): Ignored.

        Returns:
            torch.Tensor: The output tensor.
        """
        return hopfield_activation(x, self.prefer)

def hopfield_activation(x: torch.Tensor, prefer:int):
    """
    The functional version of the Hopfield activation function.

    Args:
        x (torch.Tensor): The input tensor.
        prefer (int): The prefered sign for when the input is 0.
    
    Returns:
        torch.Tensor: The output tensor.

    Raises:
        AssertionError: If prefer is not -1 or 1
    """
    assert prefer in [-1, 1], "prefer must be -1 or 1"

    if prefer == -1:
        x = torch.clamp(x, min=0.0)
        x = torch.sign(x) 
    else:
        x = torch.clamp(x, max=0.0)
        x = torch.sign(x)
        x += 1.0
        
    x = binary_to_spin(x) 
    return x