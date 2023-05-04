import torch
import torch.nn.functional as F
from hopnet.utils.transforms import mask_center_row, mask_center_column
from hopnet.utils.transforms import GaussianNoise
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


def evaluate(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()
        
        total_loss = 0.0
        total_acc = torch.zeros(3, device=device)

        for batch_idx, (images, y) in enumerate(data_loader):
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out = model(x)
            total_loss += criterion(out, target).item()
            total_acc += topk_accuracy(out, target, (1,3,5))
        
        return total_loss/len(data_loader), total_acc/len(data_loader)


def evaluate_noise(model, dataset, batch_size, loss_fn=F.l1_loss, flatten=False, device=torch.device("cpu")):
    model.eval()
    dataset.apply_transform()
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    total_loss = 0.0
    total_energy = 0.0
    noiser = GaussianNoise(mean=0.0, std=0.1)

    for batch_idx, (images, y) in enumerate(data_loader):
        x = images.to(device)
        if flatten:
            x = torch.flatten(x, start_dim=1)

        target = x.clone()

        x = noiser(x)

        out = model(x)
        total_loss += loss_fn(target, out, reduction='sum').item()
        total_energy += model.calc_energy(out).mean().item()
    

    return total_loss/len(dataset), total_energy/len(dataset)


def evaluate_mask(model, dataset, batch_size, width=0.2, loss_fn=F.l1_loss, flatten=False, device=torch.device("cpu")):
    model.eval()
    dataset.apply_transform()
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    total_loss = 0.0

    for batch_idx, (images, y) in enumerate(data_loader):
        x = images.to(device)
        if flatten:
            x = torch.flatten(x, start_dim=1)
        
        target = x.clone()
        
        x1 = mask_center_column(x, width)
        x2 = mask_center_row(x, width)
        x3 = mask_center_column(x2, width)

        y1 = model(x1)
        y2 = model(x2)
        y3 = model(x3)

        total_loss += (loss_fn(target, y1, reduction='sum').item() + loss_fn(target, y2, reduction='sum').item() + loss_fn(target, y3, reduction='sum').item()) / 3.0
    
    return total_loss / len(dataset)