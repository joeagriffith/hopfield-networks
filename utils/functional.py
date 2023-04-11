import torch
from abc import ABC, abstractmethod

# ===================================== Utils =====================================
def binary_to_spin(x):  # convert from [0, 1] to [-1, 1]
    return x * 2.0 - 1.0


# ===================================== Activation =====================================
def fast_hopfield_activation(x: torch.Tensor, prefer:int):
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


def stochastic_hopfield_activation(x: torch.Tensor, temperature:float, step_i:int):
    temperature = temperature/(2*step_i+1)
    if temperature == 0.0:
        x = torch.sign(x)
        x = x/2.0 + 0.5
    else:
        x = torch.sigmoid(x / temperature)

    x = torch.bernoulli(x)
    x = binary_to_spin(x)
    return x


# ===================================== Energy =====================================
def lyapunov_energy(x, weight, b=None):
    a = (weight * torch.bmm(x.unsqueeze(2), x.unsqueeze(1))).sum(dim=(1, 2))
    b = (x @ b).sum(dim=1) if b is not None else 0
    return -0.5 * a - b


def error_energy(x, weight, bias=None, actv_fn=None):
    next_x = x @ weight
    if bias is not None:
        next_x += bias
    if actv_fn is not None:
        next_x = actv_fn(next_x)

    return (x - next_x).abs().sum(dim=1)
    

# ===================================== Data Augmentation =====================================
def mask_center_column(image, width):
    image = image.clone()
    image[:, image.shape[1] // 2 - int(image.shape[1] * width) // 2 : image.shape[1] // 2 + int(image.shape[1] * width) // 2] = -1.0
    return image


def mask_center_row(image, width):
    image = image.clone()
    image[image.shape[0] // 2 - int(image.shape[0] * width) // 2 : image.shape[0] // 2 + int(image.shape[0] * width) // 2, :] = -1.0
    return image


def add_gaussian_noise(image, mean=0.0, std=0.001):
    noise = (torch.randn(image.shape) * std + mean)
    if image.is_cuda:
        noise = noise.to(torch.device("cuda"))
    return torch.clip(image + noise, min=0.0, max=1.0)

