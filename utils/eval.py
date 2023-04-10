import torch
import torch.nn.functional as F
from utils.functional import RandomGaussianNoise

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        device = "cuda" if output.is_cuda else "cpu"
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros(len(topk), dtype=float, device=device)
        for i, k in enumerate(topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[i] = correct_k.mul_(100.0 / batch_size)
        return res


def evaluate_denoise(model, data_loader, criterion, device, flatten = False, steps=None):
    with torch.no_grad():
        model.eval()
        loss = 0.0
        energy = 0.0
        noiser = RandomGaussianNoise(mean=0.0, std=0.1)

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = x.clone()
            x = noiser(x)

            out = model(x, steps=steps)
            loss += criterion(out, target).item()
            energy += model.calc_energy(out).mean().item()
        
        loss /= len(data_loader)
        energy /= len(data_loader)

        return loss, energy


def evaluate(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()
        
        loss = 0.0
        acc = torch.zeros(3, device=device)

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            loss += criterion(out, target).item()
            acc += topk_accuracy(out, target, (1,3,5))
        
        loss /= len(data_loader)
        acc /= len(data_loader) 

        return loss, acc
