import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from utils.functional import RandomGaussianNoise, mask_center_row, mask_center_column
from utils.eval import evaluate_denoise

def untrain_grad(model, x, loss_fn=F.l1_loss):
    y = model(x)
    loss = loss_fn(y, x)

    # scales loss between 0 and 1 (for -1 and 1 activations)
    if loss_fn == F.l1_loss:
        loss = loss / 2.0
    elif loss_fn == F.mse_loss:
        loss = loss / 4.0

    loss = 1 - (loss + 1).pow(-2.5) # loss stays high longer, from 1 to 0.

    grad = torch.bmm(y.unsqueeze(2), y.unsqueeze(1)) * loss
    grad = (torch.triu(grad, diagonal=1) + torch.tril(grad, diagonal=-1).transpose(1, 2)) / 2.0

    return grad

def train_reconstruct(
    model, 
    train_dataset,
    optimiser,
    model_name, 
    num_epochs, 
    scheduler=None,
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    save_model=True,
    error=False,
    batch_size=100,
    learning_rate=3e-4,
    untrain_after=None,
    untrain_loss_fn=F.l1_loss,
    validate_every=None,
    mode="default",
    device="cpu",
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_energy = []
    train_loss = []

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        # loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    
        loop = enumerate(train_loader)

        epoch_train_energy = 0.0

        for batch_idx, (images, y) in loop:

            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            
            energy = model.calc_energy(x, error=error).mean().abs()

            optimiser.zero_grad()
            if mode == "default":
                grad = -torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
                grad = torch.triu(grad, diagonal=1)
                model.W_upper.grad = grad.sum(dim=0)

            elif mode == "iterative":
                grad = -torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

                next_x = model.step(x)
                multiplier = x * next_x * -1 # 1 if incorrect, -1 if correct
                multiplier = (multiplier + 1) / 2 # 1 if incorrect, 0 if correct
                multiplier = multiplier.unsqueeze(2).repeat(1, 1, x.shape[1])

                grad = grad * multiplier # 0 if correct, x_i * x_j if incorrect
                grad = (torch.triu(grad, diagonal=1) + torch.tril(grad, diagonal=-1).transpose(1, 2)) / 2

                model.W_upper.grad = grad.sum(dim=0)

                # Untraining, reduces spurious minima.
                if untrain_after is not None and epoch >= untrain_after:
                    grad = untrain_grad(model, x, F.l1_loss)
                    model.W_upper.grad += grad.mean(dim=0)

            elif mode == "energy":
                energy.backward()

            optimiser.step()
            
            with torch.no_grad():
                epoch_train_energy += energy.item()


        train_energy.append(epoch_train_energy / len(train_loader))
        if validate_every is not None and epoch % validate_every == 0:
            train_loss.append(validate(model, train_dataset, loss_fn=F.l1_loss, flatten=flatten, device=device))

        if scheduler is not None:
            scheduler.step(energy)
        
        if save_model:
            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')

        step += len(train_dataset)
        writer.add_scalar("Training Energy", train_energy[-1], step)
        writer.add_scalar("Training Loss", train_loss[-1], step)
        
    return torch.tensor(train_energy), torch.tensor(train_loss), step


def validate(model, train_dataset, loss_fn=F.l1_loss, flatten=False, device=torch.device("cpu")):
    model.eval()
    train_dataset.apply_transform()
    train_loader = DataLoader(train_dataset, 1, shuffle=False)

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
