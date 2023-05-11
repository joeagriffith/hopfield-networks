import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation
from hopnet.energies import Energy

"""
The standard Hopfield Network. Weights are symmetric and there are no self connections. This is enforced by usage of the weight_sym_upper property.
Any of the implemented activation and energy functions can be used. Although Tanh may yield unpredictable results as doesnt return discrete values, but continuous ones.
this is the only model where the train_hopfield function from hopnet/utils/train.py can be used effectively.
The model trains best using the 'energy' training mode with the Error energy function and Stochastic Hopfield activation.
"""
class HopfieldNet(nn.Module):
    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, symmetric=True):
        super(HopfieldNet, self).__init__()
        self.size = size
        self.steps = steps
        self.symmetric = symmetric
        
        # weight initialisation
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.energy_fn = energy_fn
        self.actv_fn = actv_fn


    # Ensures symmetry
    @property
    def weight_sym_upper(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x, step_i):
        x =  x @ self.weight_sym_upper if self.symmetric else x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        x = self.actv_fn(x, step_i)
        return x


    # Performs multiple steps of the Hopfield network
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for i in range(steps):
            x = self.step(x, i)

        return x


    # Calculates the energy of the Hopfield network
    # error: If True, uses alternative energy function
    def calc_energy(self, x):
        return self.energy_fn(x, self.weight_sym_upper, self.bias)



        