import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FastHopfieldActivation, Activation


class HopfieldNet(nn.Module):
    def __init__(self, size: int, actv_fn:Activation, bias=False, steps=10, threshold=0.0):
        super(HopfieldNet, self).__init__()
        self.size = size
        self.steps = steps
        self.threshold = threshold
        
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.actv_fn = actv_fn


    # Ensures symmetry
    @property
    def weight_sym_upper(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x):
        x =  x @ self.weight_sym_upper # (batch_size, size) @ (size, size) = (batch_size, size)
        x = self.actv_fn(x)
        return x


    # Performs multiple steps of the Hopfield network
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for _ in range(steps):
            x = self.step(x)

        return x


    # Calculates the energy of the Hopfield network
    # error: If True, uses alternative energy function
    def calc_energy(self, x, error=False):
        if error:
            next_x = torch.tanh(x @ self.weight_sym_upper)
            return (x - next_x).abs().sum(dim=1)

        a = (self.weight_sym_upper * (torch.bmm(x.unsqueeze(2), x.unsqueeze(1)))).abs().sum(dim=(1, 2))
        b = torch.matmul(x, self.bias).abs() if self.bias is not None else 0
        return 0.5 * a + b



        