import torch
from hopnet.energies import Energy

"""
This is the standard energy function used in Hopfield networks.
While not known as the Lyapunov energy, it is a Lypunov function, which is why it has this name.
This function cannot be used with the "energy" training mode as it is not lower bounded 
and will cause the weights to explode in the negative direction.
"""
class LyapunovEnergy(Energy):
    def __call__(self, x, weight, bias):
        return lyapunov_energy(x, weight, bias)

def lyapunov_energy(x, weight, b=None):
    a = (weight * torch.bmm(x.unsqueeze(2), x.unsqueeze(1))).sum(dim=(1, 2))
    b = (x @ b) if b is not None else 0
    return -0.5 * a - b