import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functional import F_fast_hopfield_activation, F_stochastic_hopfield_activation


class HopfieldNet(nn.Module):
    def __init__(self, size: int, mode='stochastic', bias=False, init_weights=True, steps=10, temperature:float=None, prefer:int=None):
        super(HopfieldNet, self).__init__()
        assert mode in ['stochastic', 'fast'], "mode must be 'stochastic' or 'fast'"
        if mode == 'stochastic':
            assert temperature is not None, "temperature must be specified for stochastic mode"
            assert temperature >= 0.0, "temperature must be non-negative"
        elif mode == 'fast':
            assert prefer is not None, "prefer must be specified for fast mode"
            assert prefer in [-1, 1], "prefer must be -1 or 1"

        self.mode = mode
        self.size = size
        self.steps = steps
        self.temperature = temperature
        self.prefer = prefer
        
        # self.Weight = nn.Parameter(torch.rand(size, size))
        self.W_upper = nn.Parameter(torch.zeros(size, size))
        if init_weights:
            self.W_upper.data = torch.nn.init.xavier_uniform_(self.W)
            self.W_upper.data = torch.triu(self.W, diagonal=1) # Set all values at and below the diagonal to zero

        self.bias = nn.Parameter(torch.randn(size)) if bias else None


    # Ensures symmetry
    @property
    def W(self):
        return self.W_upper + self.W_upper.t()


    # Performs one step of the Hopfield network
    def step(self, x, temperature=0.0):
        x =  x @ self.W # (batch_size, size) @ (size, size) = (batch_size, size)
        if self.mode == 'stochastic':
            x = F_stochastic_hopfield_activation(x, temperature)
        else:
            x = F_fast_hopfield_activation(x, self.prefer)
        return x


    # Performs multiple steps of the Hopfield network
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for i in range(steps):
            temperature = None
            if self.mode == 'stochastic':
                temperature = self.temperature/(2*i+1)
            x = self.step(x, temperature)

        return x


    # Calculates the energy of the Hopfield network
    # error: If True, uses alternative energy function
    def calc_energy(self, x, error=False):
        if error:
            next_x = torch.tanh(x @ self.W)
            return (x - next_x).abs().sum(dim=1)

        a = (self.W * (torch.bmm(x.unsqueeze(2), x.unsqueeze(1)))).sum(dim=(1, 2))
        b = x @ self.bias if self.bias is not None else 0
        return -0.5 * a - b





        