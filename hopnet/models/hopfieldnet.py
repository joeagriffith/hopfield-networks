import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation
from hopnet.energies import Energy

class HopfieldNet(nn.Module):
    """
    The standard Hopfield Network. Weights are symmetric and there are no self connections. This is enforced by usage of the weight_sym_upper property.
    Any of the implemented activation and energy functions can be used. Although Tanh may yield unpredictable results as doesnt return discrete values, but continuous ones.
    this is the only model where the train_hopfield function from hopnet/utils/train.py can be used effectively.
    The model trains best using the 'energy' training mode with the Error energy function and Stochastic Hopfield activation.

    Args:
        size (int): The number of neurons in the network.
        energy_fn (Energy): The energy function to use.
        actv_fn (Activation): The activation function to use.
        bias (bool): Whether or not to use a bias vector.
        steps (int): The number of steps to perform when forward is called.
        symmetric (bool): Whether or not to enforce symmetry on the weights.
    """
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


    @property
    def weight_sym_upper(self):
        """
        Used to ensure symmetry. This uses the upper triangular part of the weight matrix and reflects it to the lower triangular part, leaving the diagonal as 0.

        Returns:
            torch.Tensor: A symmetric wieght matrix formulated from the upper triangular part of the weight matrix (excluding the diagonal).
        """
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    def step(self, x, step_i):
        """
        Performs one step of the Hopfield network.

        Args:
            x (torch.Tensor): The current state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
            step_i (int): The current step number. Used for stochastic activation functions.

        Returns:
            torch.Tensor: The new state of the network after one step. A 2D tensor of shape (batch_size, N).
        """
        x =  x @ self.weight_sym_upper if self.symmetric else x @ self.weight
        if self.bias is not None:
            x = x + self.bias
        x = self.actv_fn(x, step_i)
        return x


    def forward(self, x, steps=None):
        """
        Performs multiple steps of the Hopfield network.

        Args:
            x (torch.Tensor): The initial state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.
            steps (int): The number of steps to perform. If None, uses the number of steps specified in the constructor.

        Returns:
            torch.Tensor: The final state of the network after the specified number of steps. A 2D tensor of shape (batch_size, N).
        """
        if steps is None:
            steps = self.steps

        for i in range(steps):
            x = self.step(x, i)

        return x


    def calc_energy(self, x):
        """
        Calculates the energy of the Hopfield network using the specified energy function.

        Args:
            x (torch.Tensor): The state of the network. Must be a 2D tensor (batch_size, N) where N is the number of neurons in the network.

        Returns:
            torch.Tensor: The energy of the network. A 1D tensor of shape (batch_size,).
        """
        return self.energy_fn(x, self.weight_sym_upper, self.bias)



        