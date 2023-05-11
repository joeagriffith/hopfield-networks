import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation, Tanh
from hopnet.energies import Energy

"""
This model adds a second set of weights and biases to PCHNet, however no extra error neurons are added.
The new weights are used to propagate the error signal in its update step. 
This allows error neurons to affect all other neurons in the network, not just their corresponding state neurons.
As the new weights are used after the error signal is calculated, the network cannot be trained using the 'energy' training mode.
Instead, the 'reconstruction_err' must be used inorder to propagate gradients to all weights in the network.
"""

class PCHNetV2(nn.Module):
    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, threshold=0.0, eta=1.0, mu=1.0, pred_actv_fn=torch.tanh, symmetric=True):
        super(PCHNetV2, self).__init__()
        self.size = size
        self.steps = steps
        self.threshold = threshold
        
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.weight2 = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight2)
        self.bias2 = nn.Parameter(torch.randn(size)) if bias else None

        self.energy_fn = energy_fn
        self.actv_fn = actv_fn

        self.mu = mu
        self.eta = eta
        self.pred_actv_fn = pred_actv_fn
        self.symmetric = symmetric


    # Ensures symmetry
    @property
    def weight_sym_upper(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x, step_i, actv_fn=None):
        if actv_fn is None:
            actv_fn = self.actv_fn

        pred = x @ self.weight_sym_upper if self.symmetric else x @ self.weight
        if self.bias is not None:
            pred += self.bias
        if self.pred_actv_fn is not None:
            pred = self.pred_actv_fn(pred)

        # Error signal is propagated through the new set of weights, though no activation function is applied as it was found to be suboptimal.
        e = x - pred
        update = e @ self.weight2
        if self.bias2 is not None:
            update += self.bias2

        x = self.eta * x - self.mu*update
        
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

    def calc_energy(self, x):
        out, e = self.step(x, 0)
        out, e = self.step(x, 1)
        return e.square().mean()