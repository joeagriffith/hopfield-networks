from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        return add_gaussian_noise(img, self.mean, self.std)

# scales input from [min, max] to [-1, 1]
class Scale(torch.nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Scale, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (x - self.min) / (self.max - self.min) * 2.0 - 1.0

    def inverse(self, x):
        return (x + 1.0) / 2.0 * (self.max - self.min) + self.min

# ===================================== Functional =====================================
def mask_center_column(image, width):
    image = image.clone()
    image[:, :, :, image.shape[3] // 2 - int(image.shape[3] * width) // 2 : image.shape[3] // 2 + int(image.shape[3] * width) // 2] = -1.0
    return image


def mask_center_row(image, width):
    image = image.clone()
    image[:, :, image.shape[2] // 2 - int(image.shape[2] * width) // 2 : image.shape[2] // 2 + int(image.shape[2] * width) // 2, :] = -1.0
    return image

# Only for continuous images
def add_gaussian_noise(image, mean=0.0, std=0.001):
    noise = (torch.randn(image.shape) * std + mean)
    if image.is_cuda:
        noise = noise.to(torch.device("cuda"))
    return torch.clip(image + noise, min=-1.0, max=1.0)

# Only for binary images
def add_salt_and_pepper_noise(image, p=0.05):
    # noise[i, j] = -1, 0, or 1
    noise = torch.bernoulli(torch.full(image.shape, 1.0-p)) * 2.0 - 1.0
    if image.is_cuda:
        noise = noise.to(torch.device("cuda"))
    return torch.clip(image * noise, min=-1.0, max=1.0)

def downsample_and_upsample(image, scale=2):
    image = F.interpolate(image, scale_factor=1.0 / scale, mode="nearest")    
    image = F.interpolate(image, scale_factor=scale, mode="nearest")
    return image