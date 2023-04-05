import torch
import torch.nn as nn
import utils.actv_funcs.hopfield_activation as hopfiel_activation


class HopfieldNetwork(nn.Module):
    def __init__(self, size: int, bias=False, steps=10):
        super(HopfieldNetwork, self).__init__()
        self.size = size
        self.steps = steps
        
        self.W_upper = nn.Parameter(torch.randn(size, size))
        self.W_upper = torch.nn.init.xavier_uniform_(self.W)
        self.W_upper = torch.triu(self.W, diagonal=1) # Set all values at and below the diagonal to zero

        self.bias = nn.Parameter(torch.randn(size)) if bias else None


    # Ensures symmetry
    @property
    def W(self):
        return self.W_upper + self.W_upper.t()


    def step(self, x):
        x =  x @ self.W # (batch_size, size) @ (size, size) = (batch_size, size)
        x = hopfield_activation(x)
        return x

         


    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps

        for _ in range(steps):
            x = self.step(x)

        return x


    def calc_energy(self, x):
        a = (self.W * (torch.bmm(x.unsqueeze(2), x.unsqueeze(1)))).sum(dim=(1, 2))
        b = torch.matmul(x, self.bias) if self.bias is not None else 0
        return -0.5 * a - b



        