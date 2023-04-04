import torch
import torch.nn as nn


class HopfieldNetwork(nn.Module):
    def __init__(self, size: int, bias=False, steps=10):
        super(HopfieldNetwork, self).__init__()
        self.size = size
        self.steps = steps
        self.linear = nn.Linear(size, size, bias=bias)
        # remove self-connections
        self.zero_diagonal()


    def zero_diagonal(self):
        self.linear.weight.data.fill_diagonal_(0)


    def step(self, x):
        return torch.sign(self.linear(x))



    def forward(self, x, steps=None, detach=False):
        if steps is None:
            steps = self.steps

        for i in range(steps):
            x = self.step(x)

            if detach:
                x = x.detach()

        return x


    def calc_energy(self, x):
        a = (self.linear.weight * (torch.bmm(x.unsqueeze(2), x.unsqueeze(1)))).sum(dim=(1, 2))
        b = torch.matmul(x, self.linear.bias) if self.linear.bias is not None else 0
        return -0.5 * a - b



        