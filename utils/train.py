import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from utils.eval import evaluate_noise, evaluate_mask
    
def untrain_grad(x, model, optimiser, mode, loss_fn=F.l1_loss, untrain_const=0.5):
    y = model(x)
    loss = loss_fn(y, x, reduce=False).mean(dim=1)

    # scales loss between 0 and 1 (for -1 and 1 activations)
    if loss_fn == F.l1_loss:
        loss = loss / 2.0
    elif loss_fn == F.mse_loss:
        loss = loss / 4.0

    loss = 1 - (loss + 1).pow(-2.5) # loss stays high longer, from 1 to 0.

    if mode == 'default' or mode == 'gardiner':
        grad = torch.bmm(y.unsqueeze(2), y.unsqueeze(1)) * loss.view(-1, 1, 1) * untrain_const
        grad = (torch.triu(grad, diagonal=1) + torch.tril(grad, diagonal=-1).transpose(1, 2)) / 2.0
        if model.weight.grad is None:
            model.weight.grad = grad.mean(dim=0)
        else:
            model.weight.grad += grad.mean(dim=0)
    elif mode == 'energy':
        energy = -model.calc_energy(y) * loss * untrain_const
        energy.mean().backward()


def train_denoise(
    model, 
    train_dataset,
    optimiser,
    scheduler,
    model_name, 
    num_epochs, 
    mode='default',
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    save_model=True,
    batch_size=100,
    untrain_after=None,
    untrain_loss_fn=F.l1_loss,
    untrain_const=0.5,
    validate_every=None,
    device="cpu",
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_loss = []
    train_energy = []
    
    best_train_loss = float("inf")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    assert mode in ['default', 'gardiner', 'energy']

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        epoch_train_energy = 0.0

        for batch_idx, (images, y) in loop:
            if epoch > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_energy = train_energy[-1],
                    train_loss = train_loss[-1], 
                )

            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)

            optimiser.zero_grad()

            energy = model.calc_energy(x).mean()

            if mode == 'default':
                grad = -torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
                grad = torch.triu(grad, diagonal=1)

                if model.weight.grad is None:
                    model.weight.grad = grad.mean(dim=0)
                else:
                    model.weight.grad += grad.mean(dim=0)
                
                if model.bias is not None:
                    if model.bias.grad is None:
                        model.bias.grad = x.mean(dim=0)
                    else:
                        model.bias.grad += x.mean(dim=0)

            #  May be unsuitable name 
            elif mode == 'gardiner':
                grad = -torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

                next_x = model.step(x, 0)
                multiplier = x * next_x # 1 if correct, -1 if incorrect
                multiplier = (-multiplier + 1) / 2 # 0 if correct, 1 if incorrect
                # repeat for each inp_node
                mat_multiplier = multiplier.unsqueeze(1).repeat(1, x.shape[1], 1)

                grad = grad * mat_multiplier # 0 if correct, -x*x if incorrect
                grad = (torch.triu(grad, diagonal=1) + torch.tril(grad, diagonal=-1)) / 2.0

                if model.weight.grad is None:
                    model.weight.grad = grad.mean(dim=0)
                else:
                    model.weight.grad += grad.mean(dim=0)
                
                if model.bias is not None:
                    if model.bias.grad is None:
                        model.bias.grad = (x*multiplier).mean(dim=0)
                    else:
                        model.bias.grad += (x*multiplier).mean(dim=0)

            elif mode == 'energy':
                energy.backward()
            
            # Untrain to reduce spurious minima
            if untrain_after is not None and epoch > untrain_after:
                untrain_grad(x, model, optimiser, mode, untrain_loss_fn, untrain_const)

            optimiser.step()
            
            with torch.no_grad():
                epoch_train_energy += energy.item()


        train_energy.append(epoch_train_energy / len(train_loader))

        if validate_every is not None:
            if epoch == 0 or (epoch+1) % validate_every == 0:
                with torch.no_grad():
                    train_loss.append(evaluate_mask(model, train_dataset, batch_size=4, loss_fn=F.l1_loss, flatten=flatten, device=device))
                    if scheduler is not None:
                        scheduler.step(train_loss[-1])
            
                if save_model:
                    if best_train_loss > train_loss[-1]:
                        torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
                        writer.add_scalar("Training Loss", train_loss[-1], step)

        step += len(train_dataset)
        writer.add_scalar("Training Energy", train_energy[-1], step)
        
    return torch.tensor(train_energy), torch.tensor(train_loss), step
