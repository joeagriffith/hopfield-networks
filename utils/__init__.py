from abc import ABC, abstractmethod
from utils.functional import fast_hopfield_activation, stochastic_hopfield_activation, lyapunov_energy, error_energy, add_gaussian_noise

#  ===================================== Activation =====================================
class Activation(ABC):
    @abstractmethod
    def __call__(self, x, step_i):
        pass

class FastHopfieldActivation(Activation):
    def __init__(self, prefer:int):
        assert prefer in [-1, 1], "prefer must be -1 or 1"
        self.prefer = prefer

    def __call__(self, x, _=None):
        return fast_hopfield_activation(x, self.prefer)

class StochasticHopfieldActivation(Activation):
    def __init__(self, temperature:float):
        self.temperature = temperature

    def __call__(self, x, step_i:int):
        return stochastic_hopfield_activation(x, self.temperature, step_i)


# ===================================== Energy =====================================
class Energy(ABC):
    @abstractmethod
    def __call__(self, x, weight, bias):
        pass

class LyapunovEnergy(Energy):
    def __call__(self, x, weight, bias):
        return lyapunov_energy(x, weight, bias)

class ErrorEnergy(Energy):
    def __init__(self, actv_fn=None):
        self.actv_fn = actv_fn

    def __call__(self, x, weight, bias=None):
        return error_energy(x, weight, bias, self.actv_fn)


# ===================================== Data Augmentation =====================================
class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        return add_gaussian_noise(img, self.mean, self.std)