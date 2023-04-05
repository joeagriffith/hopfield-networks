import torch

def hopfield_activation(x: torch.Tensor):
    x = torch.clamp(x, min=0)
    x = torch.sign(x)
    x = x * 2 - 1
    return x