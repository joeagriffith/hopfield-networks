import torch
from hopnet.activations import Activation
from hopnet.utils.functional import binary_to_spin

class FastHopfieldActivation(Activation):
    def __init__(self, prefer:int):
        assert prefer in [-1, 1], "prefer must be -1 or 1"
        self.prefer = prefer

    def __call__(self, x, _=None):
        return fast_hopfield_activation(x, self.prefer)

def fast_hopfield_activation(x: torch.Tensor, prefer:int):
    assert prefer in [-1, 1], "prefer must be -1 or 1"

    if prefer == -1:
        x = torch.clamp(x, min=0.0)
        x = torch.sign(x) 
    else:
        x = torch.clamp(x, max=0.0)
        x = torch.sign(x)
        x += 1.0
        
    x = binary_to_spin(x) 
    return x