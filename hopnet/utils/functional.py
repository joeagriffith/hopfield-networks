from tkinter import Image

import numpy as np
from abc import ABC, abstractmethod

# ===================================== Utils =====================================
def binary_to_spin(x):  # convert from [0, 1] to [-1, 1]
    """
    Used to convert tensors from binary to spin representation.
    [0, 1] -> [-1, 1]

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    return x * 2.0 - 1.0




    
