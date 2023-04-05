import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.extra_funcs import HopfieldActivation


class HopfieldNetwork(nn.Module):
    def __init__(self, size: int, bias=False, steps=10, threshold=0.0):
        super(HopfieldNetwork, self).__init__()
        self.size = size
        self.steps = steps
        self.threshold = threshold
        
        # self.Weight = nn.Parameter(torch.rand(size, size))
        self.W_upper = nn.Parameter(torch.zeros(size, size))
        self.W_upper.data = torch.nn.init.xavier_uniform_(self.W)
        self.W_upper.data = torch.triu(self.W, diagonal=1) # Set all values at and below the diagonal to zero

        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.hopfield_activation = HopfieldActivation(threshold=threshold)


    # Ensures symmetry
    @property
    def W(self):
        return self.W_upper + self.W_upper.t()


    def step(self, x):
        x =  x @ self.W # (batch_size, size) @ (size, size) = (batch_size, size)
        x = self.hopfield_activation(x)
        return x


    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for _ in range(steps):
            x = self.step(x)

        return x


    def calc_energy(self, x, error=False):
        if error:
            next_x = torch.tanh(x @ self.W)
            return (x - next_x).abs().sum(dim=1)

        a = (self.W * (torch.bmm(x.unsqueeze(2), x.unsqueeze(1)))).abs().sum(dim=(1, 2))
        b = torch.matmul(x, self.bias).abs() if self.bias is not None else 0
        return 0.5 * a + b



        