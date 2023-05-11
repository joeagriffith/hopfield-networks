from abc import ABC, abstractmethod

# Abstract base class for energy functions, inherit from this class to create a new activation function
class Energy(ABC):
    @abstractmethod
    def __call__(self, x, weight, bias):
        pass

from hopnet.energies.lyapunov import LyapunovEnergy, lyapunov_energy
from hopnet.energies.error import ErrorEnergy, error_energy