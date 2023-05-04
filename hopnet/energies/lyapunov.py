import torch
from hopnet.energies import Energy

class LyapunovEnergy(Energy):
    def __call__(self, x, weight, bias):
        return lyapunov_energy(x, weight, bias)

def lyapunov_energy(x, weight, b=None):
    a = (weight * torch.bmm(x.unsqueeze(2), x.unsqueeze(1))).sum(dim=(1, 2))
    b = (x @ b) if b is not None else 0
    return -0.5 * a - b