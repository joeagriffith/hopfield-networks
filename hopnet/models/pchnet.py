import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation
from hopnet.energies import Energy


class PCHNet(nn.Module):
    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, threshold=0.0, eta=1.0, mu=1.0, pred_actv_fn=torch.tanh):
        super(PCHNet, self).__init__()
        self.size = size
        self.steps = steps
        self.threshold = threshold
        
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.energy_fn = energy_fn
        self.actv_fn = actv_fn

        self.eta = eta
        self.mu = mu
        self.pred_actv_fn = pred_actv_fn


    # Ensures symmetry
    @property
    def weight_sym_upper(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x, step_i, actv_fn=None):
        if actv_fn is None:
            actv_fn = self.actv_fn

        pred = x @ self.weight_sym_upper
        if self.bias is not None:
            pred = pred + self.bias
        if self.pred_actv_fn is not None:
            pred = self.pred_actv_fn(pred)

        e = x - pred
        x = self.eta * x - self.mu * e
        if actv_fn is not None:
            x = actv_fn(x, step_i)

        return x, e

    # Performs multiple steps of the Hopfield network
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for i in range(steps):
            x, e = self.step(x, i)

        return x

    # Calculates the energy of the Hopfield network
    # error: If True, uses alternative energy function
    def calc_energy(self, x):
        # return self.energy_fn(x, self.weight_sym_upper, self.bias)

        out, e = self.step(x, 0)
        return e.square().mean()




        