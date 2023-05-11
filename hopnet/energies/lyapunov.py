import torch
from hopnet.energies import Energy

class LyapunovEnergy(Energy):
    """
    This is the standard energy function used in Hopfield networks.
    While not known as the Lyapunov energy, it is a Lypunov function, which is why it has this name.
    This function cannot be used with the "energy" training mode as it is not lower bounded 
    and will cause the weights to explode in the negative direction.
    """
    def __call__(self, x, weight, bias):
        """
        Calculates the energy of the network in its current state using the standard Hopfield energy function.

        Args:
            x (torch.Tensor): The current state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
            weight (torch.Tensor): The weight matrix of the network. Must be a 2D tensor (N, N).
            bias (torch.Tensor): The bias vector of the network. If None, no bias is applied. If not None, must be a 1D tensor (N).

        Returns:
            torch.Tensor: The energy of the network in its current state. A 1D tensor of length batch_size.
        """
        return lyapunov_energy(x, weight, bias)

def lyapunov_energy(x, weight, b=None):
    """
    The standard Hopfield energy function.

    Args:
        x (torch.Tensor): The current state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
        weight (torch.Tensor): The weight matrix of the network. Must be a 2D tensor (N, N).
        b (torch.Tensor): The bias vector of the network. If None, no bias is applied. If not None, must be a 1D tensor (N).

    Returns:
        torch.Tensor: The energy of the network in its current state. A 1D tensor of length batch_size.
    """
    a = (weight * torch.bmm(x.unsqueeze(2), x.unsqueeze(1))).sum(dim=(1, 2))
    b = (x @ b) if b is not None else 0
    return -0.5 * a - b