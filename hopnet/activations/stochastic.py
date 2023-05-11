import torch
from hopnet.activations import Activation
from hopnet.utils.functional import binary_to_spin

"""
This activation function implements simulated annealing.
By adding noise to the activation process we can escape local minima.
This noise is controlled by the temperature parameter.
The noise is reduced as the step number, 'step_i', increases.
This activation also works to regularise the network in training,
taking longer to converge, but often converging to a better solution.
"""
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
        x = torch.sigmoid(x / temperature) # sigmoid is used to calculate the probability of an activation being 1, as opposed to 0

    x = torch.bernoulli(x) # Samples from a bernoulli distribution to get activations of 0 or 1
    x = binary_to_spin(x) # Converts the activations to -1 or 1
    return x