import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation
from hopnet.energies import Energy

"""
Inpired by Predictive Coding, this architecture realises the error energy directly in its computation.
The model introduces error neurons which capture the systems discrepancy between the current state and the next state.
Error neurons have a one-to-one relationship with the state neurons.
As in HopfieldNet(), the weights are symmetric and there are no self connections, both of which are enforced by usage of the weight_sym_upper property.
While any activation function can be used, baring Tanh, the user is suggested to use the Error energy function.
The network cannot be trained using the train_hopfield function, and trains best using the 'energy' or 'reconstruction_err' training modes.
"""
class PCHNet(nn.Module):
    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, threshold=0.0, eta=1.0, mu=1.0, pred_actv_fn=None):
        super(PCHNet, self).__init__()
        self.size = size
        self.steps = steps
        self.threshold = threshold
        
        # weight initialisation
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.energy_fn = energy_fn
        self.actv_fn = actv_fn

        self.eta = eta # decay rate of the state neurons
        self.mu = mu # learning rate of the state neurons, using the error signal
        self.pred_actv_fn = pred_actv_fn # activation function applied to the error signal. None is suitable, though torch.tanh is acceptable.


    # Ensures symmetry
    @property
    def weight_sym_upper(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x, step_i, actv_fn=None):
        if actv_fn is None:
            actv_fn = self.actv_fn

        # calculate predictions
        pred = x @ self.weight_sym_upper
        if self.bias is not None:
            pred = pred + self.bias
        if self.pred_actv_fn is not None:
            pred = self.pred_actv_fn(pred)

        # calculate errors and update state
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




        