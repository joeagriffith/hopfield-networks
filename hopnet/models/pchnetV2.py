import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation, Tanh
from hopnet.energies import Energy


class PCHNetV2(nn.Module):
    """
    |  This model adds a second set of weights and biases to PCHNet, however no extra error neurons are added.
    |  The new weights are used to propagate the error signal in its update step. 
    |  This allows error neurons to affect all other neurons in the network, not just their corresponding state neurons.
    |  As the new weights are used after the error signal is calculated, the network cannot be trained using the 'energy' training mode.
    |  Instead, the 'reconstruction_err' must be used inorder to propagate gradients to all weights in the network.

    Args:
        |  size (int): The number of neurons in the network.
        |  energy_fn (Energy): The energy function to use.
        |  actv_fn (Activation): The activation function to use.
        |  bias (bool): Whether or not to use a bias vector.
        |  steps (int): The number of steps to perform when forward is called.
        |  eta (float): The decay rate of the state neurons.
        |  mu (float): The learning rate of the state neurons, using the error signal.
        |  pred_actv_fn (Activation): The activation function applied to the error signal. None is suitable, though torch.tanh is acceptable.
        |  symmetric (bool): Whether or not to enforce symmetry on the weights.
    """
    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, eta=1.0, mu=1.0, pred_actv_fn=torch.tanh, symmetric=True):
        super(PCHNetV2, self).__init__()
        self.size = size
        self.steps = steps
        
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


    @property
    def weight_sym_upper(self):
        """
        Used to ensure symmetry. This uses the upper triangular part of the weight matrix and reflects it to the lower triangular part, leaving the diagonal as 0.

        Returns:
            torch.Tensor: A symmetric wieght matrix formulated from the upper triangular part of the weight matrix (excluding the diagonal).
        """
        return torch.triu(self.weight, diagonal=1) + torch.triu(self.weight, diagonal=1).t()


    def step(self, x, step_i, actv_fn=None):
        """
        |  Performs a single step of the network. The network first calculates a prediction of the next state using the standard hopfield update rule.
        |  This prediction is compared against the current state of the network to calculate the error signal. 
        |  The error signal is then propagated through the second set of weights and biases to calculate the update to the state neurons.
        |  The state neurons are then updated using the update rule using a decay rate eta and learning rate mu.

        Args:
            |  x (torch.Tensor): The current state of the network. Must be a 2d tensor of shape (batch_size, size).
            |  step_i (int): The current step of the network. Used in the stochastic activation function.
            |  actv_fn (Activation): The activation function to use. If None, the default activation function is used.

        Returns:
            |  torch.Tensor: The new state of the network. A 2d tensor of shape (batch_size, size).
            |  torch.Tensor: The error signal of the network. A 2d tensor of shape (batch_size, size).
        """
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

    def forward(self, x, steps=None):
        """
        Performs a forward pass of the network. The network is iterated for the specified number of steps, or the default number of steps if none is specified.

        Args:
            |  x (torch.Tensor): The initial state of the network. Must be a 2d tensor of shape (batch_size, size).
            |  steps (int): The number of steps to perform. If None, the default number of steps is used.

        Returns:
            torch.Tensor: The final state of the network. A 2d tensor of shape (batch_size, size).
        """

        if steps is None:
            steps = self.steps

        for i in range(steps):
            x, e = self.step(x, i)

        return x

    def calc_energy(self, x):
        """
        |  Calculates the energy of the network for the given state.
        |  Two steps are performed to allow the second set of weights to be used in the calculation.
        |  The energy before the network has converged should not be compared to that of PCHNet as its takes more steps to calculate the energy.

        Args:
            x (torch.Tensor): The state of the network. Must be a 2d tensor of shape (batch_size, size).

        Returns:
            torch.Tensor: The energy of the network. A 1d tensor of shape (batch_size,).
        """
        out, e = self.step(x, 0)
        out, e = self.step(x, 1)
        return e.square().mean()