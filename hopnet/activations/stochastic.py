import torch
from hopnet.activations import Activation
from hopnet.utils.functional import binary_to_spin


class StochasticHopfieldActivation(Activation):
    def __init__(self, temperature:float):
        self.temperature = temperature

    def __call__(self, x, step_i:int):
        return stochastic_hopfield_activation(x, self.temperature, step_i)
        
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