import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldNet(nn.Module):
    def __init__(self, 
                 size:int, 
                 energy_fn, 
                 actv_fn=None, 
                 bias=False, 
                 init_weights=True, 
                 steps=10):

        super(HopfieldNet, self).__init__()

        self.size = size
        self.steps = steps

        self.actv_fn = actv_fn
        self.energy_fn = energy_fn
        
        self.weight = nn.Parameter(torch.zeros(size, size))
        if init_weights:
            torch.nn.init.xavier_uniform_(self.weight)
            self.weight.data = torch.triu(self.weight.data, diagonal=1) # Set all values at and below the diagonal to zero

        self.bias = nn.Parameter(torch.randn(size)) if bias else None


    # Completes symmetry
    @property
    def full_weight(self):
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    # Performs one step of the Hopfield network
    def step(self, x, step_i:int):
        x =  x @ self.full_weight # (batch_size, size) @ (size, size) = (batch_size, size)
        if self.bias is not None:
            x += self.bias
        if self.actv_fn is not None:
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
        return self.energy_fn(x, self.full_weight, self.bias)





        