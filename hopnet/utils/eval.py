import torch
import torch.nn.functional as F
from hopnet.utils.transforms import mask_center_row, mask_center_column, downsample_and_upsample, add_salt_and_pepper_noise
from hopnet.utils.transforms import GaussianNoise
from torch.utils.data import DataLoader


def topk_accuracy(output, target, topk=(1,)):
    """
    Computes the precision for the specified values of k
    
    Args:
        output (torch.Tensor): The output tensor.
        target (torch.Tensor): The target tensor.
        topk (tuple): The values of k to compute the precision for.

    Returns:
        torch.Tensor: The precision for the specified values of k.
    """

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


def evaluate(model, data_loader, device, flatten=False, width=0.2):
    """
    Returns the percentage of pixels that are different between the output and the raw input>
    Score is a mean across 5 noising functions.
    Noising functions are:
        - Masking the center column
        - Masking the center row
        - Masking the center row and column
        - Downsampling and upsampling
        - Adding salt and pepper noise

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader to use.
        device (str): The device to use.
        flatten (bool): Whether or not to flatten the input.
        width (float): The width of the mask to use.

    Returns:
        float: The percentage of pixels that are different between the output and the raw input.
    """

    with torch.no_grad():
        model.eval()
        
        total_loss = 0.0

        for _, (images, y) in enumerate(data_loader):
            images = images.to(device)

            x = [
                mask_center_column(images, width),
                mask_center_row(images, width),
                mask_center_row(mask_center_column(images, width), width),
                downsample_and_upsample(images, 2),
                add_salt_and_pepper_noise(images, 0.1),
            ]
            
            if flatten:
                x = [torch.flatten(x_i, start_dim=1) for x_i in x]
                images = torch.flatten(images, start_dim=1)

            out = [model(x_i) for x_i in x]
            diffs = torch.tensor([torch.ne(out_i, images).sum() for out_i in out]).float()
            total_loss += diffs.mean().item() / images.view(images.shape[0], -1).shape[1] * 100.0
        
        return total_loss/len(data_loader)

def evaluate_mask(model, dataloader, batch_size, width=0.2, loss_fn=F.l1_loss, flatten=False, device=torch.device("cpu")):
    """
    Alternative measure of loss, not scaled the same as the above and uses less noising functions.
    Noising functions are:
        - Masking the center column
        - Masking the center row
        - Masking the center row and column

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader to use.
        batch_size (int): The batch size to use.
        width (float): The width of the mask to use.
        loss_fn (torch.nn.Module): The loss function to use.
        flatten (bool): Whether or not to flatten the input.
        device (str): The device to use.

    Returns:
        float: The loss. Not scaled the same as in evaluate().

    """
    model.eval()
    total_loss = 0.0
    n = 0

    for batch_idx, (images, y) in enumerate(dataloader):
        n += images.shape[0]
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
    
    return total_loss / n