import torch
from abc import ABC, abstractmethod

# ===================================== Utils =====================================
def binary_to_spin(x):  # convert from [0, 1] to [-1, 1]
    return x * 2.0 - 1.0


# ===================================== Activation =====================================
class Activation(ABC):
    @abstractmethod
    def __call__(self, x, step_i):
        pass

class FastHopfieldActivation(Activation):
    def __init__(self, prefer:int):
        assert prefer in [-1, 1], "prefer must be -1 or 1"
        self.prefer = prefer

    def __call__(self, x, _):
        return fast_hopfield_activation(x, self.prefer)

class StochasticHopfieldActivation(Activation):
    def __init__(self, temperature:float):
        self.temperature = temperature

    def __call__(self, x, step_i:int):
        return stochastic_hopfield_activation(x, self.temperature, step_i)
    

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
class Energy(ABC):
    @abstractmethod
    def __call__(self, x, weight, bias):
        pass

class LyapunovEnergy(Energy):
    def __call__(self, x, weight, bias):
        return lyapunov_energy(x, weight, bias)

class ErrorEnergy(Energy):
    def __init__(self, actv_fn=torch.tanh):
        self.actv_fn = actv_fn

    def __call__(self, x, weight, bias):
        return error_energy(x, weight, bias, self.actv_fn)


def lyapunov_energy(x, weight, b=None):
    a = weight * torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
    b = x @ b if b is not None else 0
    return -0.5 * a - b


def error_energy(x, weight, b=None, actv_fn=torch.tanh):
    next_x = x @ weight
    if b is not None:
        next_x += b
    if actv_fn is not None:
        next_x = actv_fn(next_x)

    return (x - next_x)
    

# ===================================== Data Augmentation =====================================
def mask_center_column(image, width):
    image = image.clone()
    image[:, image.shape[1] // 2 - int(image.shape[1] * width) // 2 : image.shape[1] // 2 + int(image.shape[1] * width) // 2] = -1.0
    return image


def mask_center_row(image, width):
    image = image.clone()
    image[image.shape[0] // 2 - int(image.shape[0] * width) // 2 : image.shape[0] // 2 + int(image.shape[0] * width) // 2, :] = -1.0
    return image


class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        noise = (torch.randn(img.shape) * self.std + self.mean)
        if img.is_cuda:
            noise = noise.to("cuda")
        return torch.clip(img + noise, min=0.0, max=1.0)