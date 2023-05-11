import torch
import torch.nn as nn
import torch.nn.functional as F
from hopnet.activations import Activation
from hopnet.energies import Energy

class PCHNet(nn.Module):
    """
    |  Inpired by Predictive Coding, this architecture realises the error energy directly in its computation.
    |  The model introduces error neurons which capture the systems discrepancy between the current state and the next state.
    |  Error neurons have a one-to-one relationship with the state neurons.
    |  As in HopfieldNet(), the weights are symmetric and there are no self connections, both of which are enforced by usage of the weight_sym_upper property.
    |  While any activation function can be used, baring Tanh, the user is suggested to use the Error energy function.
    |  The network cannot be trained using the train_hopfield function, and trains best using the 'energy' or 'reconstruction_err' training modes.

    Args:
        |  size (int): The number of neurons in the network.
        |  energy_fn (Energy): The energy function to use.
        |  actv_fn (Activation): The activation function to use.
        |  bias (bool): Whether or not to use a bias vector.
        |  steps (int): The number of steps to perform when forward is called.
        |  eta (float): The decay rate of the state neurons.
        |  mu (float): The learning rate of the state neurons, using the error signal.
        |  pred_actv_fn (Activation): The activation function applied to the error signal. None is suitable, though torch.tanh is acceptable.
    """

    def __init__(self, size: int, energy_fn:Energy, actv_fn:Activation, bias=False, steps=10, eta=1.0, mu=1.0, pred_actv_fn=None):
        super(PCHNet, self).__init__()
        self.size = size
        self.steps = steps
        
        # weight initialisation
        self.weight = nn.Parameter(torch.zeros(size, size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.randn(size)) if bias else None

        self.energy_fn = energy_fn
        self.actv_fn = actv_fn

        self.eta = eta # decay rate of the state neurons
        self.mu = mu # learning rate of the state neurons, using the error signal
        self.pred_actv_fn = pred_actv_fn # activation function applied to the error signal. None is suitable, though torch.tanh is acceptable.


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
        |  Performs one step of network. The network first calculates a prediction of the next state using the standard Hopfield update rule.
        |  This prediction is compared against the current state of the network to calculate the error signal. 
        |  The error signal is then used to update the state of the network towards the prediction.
        |  This update is performed using the decay rate eta and the learning rate mu.

        Args:
            |  x (torch.Tensor): The current state of the network. Must be a 2d tensor of shape (batch_size, size).
            |  step_i (int): The current step of the network, used in the stochastic activation function.
            |  actv_fn (Activation): The activation function to use. If None, the default activation function is used.

        Returns:
            |  torch.Tensor: The new state of the network. A 2d tensor of shape (batch_size, size).
            |  torch.Tensor: The error tensor. A 2d tensor of shape (batch_size, size).
        """
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

    def forward(self, x, steps=None):
        """
        Performs multiple steps of the network.

        Args:
            |  x (torch.Tensor): The current state of the network. Must be a 2d tensor of shape (batch_size, size).
            |  steps (int): The number of steps to perform. If None, the default number of steps is used.
            
        Returns:
            torch.Tensor: The new state of the network. A 2d tensor of shape (batch_size, size).
        """
        if steps is None:
            steps = self.steps

        for i in range(steps):
            x, e = self.step(x, i)

        return x

    def calc_energy(self, x):
        """"
        Calculates the energy of the network.

        Args:
            x (torch.Tensor): The current state of the network. Must be a 2d tensor of shape (batch_size, size).

        Returns:
            torch.Tensor: The energy of the network. A 1d tensor of shape (batch_size,).
        """
        # return self.energy_fn(x, self.weight_sym_upper, self.bias)

        out, e = self.step(x, 0)
        return e.square().mean()




        