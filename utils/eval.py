import torch
import torch.nn.functional as F
from utils.functional import RandomGaussianNoise, mask_center_row, mask_center_column
from torch.utils.data import DataLoader


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


def reconstruct_score(model, train_dataset, batch_size, loss_fn=F.l1_loss, flatten=False, device=torch.device("cpu")):
    model.eval()
    train_dataset.apply_transform()
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    total_loss = 0.0

    for batch_idx, (images, y) in enumerate(train_loader):
        x = images.to(device)
        if flatten:
            x = torch.flatten(x, start_dim=1)
        
        x1 = mask_center_column(x)
        x2 = mask_center_row(x)
        x3 = mask_center_column(x2)

        y1 = model(x1)
        y2 = model(x2)
        y3 = model(x3)

        total_loss += loss_fn(y1, x1).item() + loss_fn(y2, x2).item() + loss_fn(y3, x3).item()
    
    return total_loss / len(train_loader)