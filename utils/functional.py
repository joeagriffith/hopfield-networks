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

# Possibly grad version
# class HopfieldActivationGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, threshold=0.0):
#         ctx.save_for_backward(x)
#         ctx.threshold = threshold
#         return HopfieldActivation.apply(x, threshold)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         threshold = ctx.threshold
#         grad_input = grad_output.clone()
#         grad_input[x < threshold] = 0.0
#         return grad_input, None


class HopfieldActivation(object):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    
    def __call__(self, x: torch.Tensor):
        x = torch.clamp(x, min=self.threshold)
        x = x - torch.ones_like(x) * self.threshold
        x = torch.sign(x)
        x = x * 2.0 - 1.0
        return x

# functional version
def F_hopfield_activation(x: torch.Tensor, threshold=0.0):
    x = torch.clamp(x, min=threshold)
    x = x - torch.ones_like(x) * threshold
    x = torch.sign(x)
    x = x * 2.0 - 1.0
    return x