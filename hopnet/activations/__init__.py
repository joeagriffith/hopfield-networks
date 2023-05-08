from abc import ABC, abstractmethod

class Activation(ABC):
    @abstractmethod
    def __call__(self, x, step_i):
        pass

from hopnet.activations.fast import FastHopfieldActivation, fast_hopfield_activation
from hopnet.activations.stochastic import StochasticHopfieldActivation, stochastic_hopfield_activation
from hopnet.activations.tanh import Tanh, tanh
from hopnet.activations.identity import Identity, identity