from abc import ABC, abstractmethod

# Abstract base class for activation functions, inherit from this class to create a new activation function
class Activation(ABC):
    @abstractmethod
    def __call__(self, x, step_i):
        pass

from hopnet.activations.hopfield import HopfieldActivation, hopfield_activation
from hopnet.activations.stochastic import StochasticHopfieldActivation, stochastic_hopfield_activation
from hopnet.activations.tanh import Tanh, tanh
from hopnet.activations.identity import Identity, identity