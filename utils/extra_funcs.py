import torch


class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        noise = (torch.randn(img.shape) * self.std + self.mean)
        if img.is_cuda:
            noise = noise.to("cuda")
        return torch.clip(img + noise, min=0.0, max=1.0)


class FastHopfieldActivation():
    def __init__(self, prefer:int):
        assert prefer in [-1, 1], "prefer must be -1 or 1"
        self.prefer = prefer

    def __call__(self, x):
        return F_fast_hopfield_activation(x, self.prefer)

# convert from [0, 1] to [-1, 1]
def binary_to_spin(x):
    return x * 2.0 - 1.0
    

def F_fast_hopfield_activation(x: torch.Tensor, prefer:int):
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


def F_stochastic_hopfield_activation(x: torch.Tensor, temperature:float):
    if temperature == 0.0:
        x = torch.sign(x)
        x = x/2.0 + 0.5
    else:
        x = torch.sigmoid(x / temperature)

    x = torch.bernoulli(x)
    x = binary_to_spin(x)
    return x


def mask_center_column(image, width=1.0):
    image = image.clone()
    image[:, image.shape[1] // 2 - int(image.shape[1] * width) // 2 : image.shape[1] // 2 + int(image.shape[1] * width) // 2] = -1.0
    # image[2 * image.shape[0] // 3:, :] = -1.0
    return image

def mask_center_row(image, width=1.0):
    image = image.clone()
    image[image.shape[0] // 2 - int(image.shape[0] * width) // 2 : image.shape[0] // 2 + int(image.shape[0] * width) // 2, :] = -1.0
    return image