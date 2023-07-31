import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from hopnet.utils.eval import evaluate_mask, evaluate
from hopnet.models import PCHNet, PCHNetV2
from hopnet.activations import Tanh
    

def untrain_grad(x, model, optimiser, mode, loss_fn=F.l1_loss, untrain_const=0.5):
    """
    |  Increases the energy at the output of the network proportionally to the reconstruction loss.
    |  Aims to combat the effects of spurious minima, by directly increasing their energy.

    Args:
        |  x (torch.Tensor): The input tensor.
        |  model (torch.nn.Module): The model to evaluate.
        |  optimiser (torch.optim.Optimizer): The optimiser to use.
        |  mode (str): The training mode to use.
        |  loss_fn (torch.nn.Module): The loss function to use.
        |  untrain_const (float): The constant to multiply the gradient by.
    """
    y = model(x)
    loss = loss_fn(y, x, reduction='none').mean(dim=1)

    # scales loss between 0 and 1 (for -1 and 1 activations)
    if loss_fn == F.l1_loss:
        loss = loss / 2.0
    elif loss_fn == F.mse_loss:
        loss = loss / 4.0

    loss = 1 - (loss + 1).pow(-2.5) # Pushes the loss higher for smaller values, still between 0 and 1

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

def train_hopfield(
    model,
    train_loader,
    model_name,
    flatten=False,
    model_dir="out/weights",
    log_dir="out/logs",
    save_model=True,
    device="cpu",
):
    """
    |  Only to be used with the HopfieldNet model. 
    |  This is the standard learning procedure for Hopfield Networks
    |  which is to take the outer product of the data with itself and averages over the dataset.
    |  This is a one-step method.

    Args:
        |  model (torch.nn.Module): The model to train.
        |  train_loader (torch.utils.data.DataLoader): The training data.
        |  model_name (str): The name of the model.
        |  flatten (bool): Whether to flatten the input data.
        |  model_dir (str): The directory to save the model weights.
        |  log_dir (str): The directory to save the logs.
        |  save_model (bool): Whether to save the model weights.
        |  device (str): The device to use.
    """
    weight = torch.zeros_like(model.weight)
    bias = torch.zeros_like(model.bias)

    for batch_idx, (images, y) in enumerate(train_loader):
        x = images.to(device)
        if flatten:
            x = x.view(x.shape[0], -1)


        weight += torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).mean(dim=0)
        bias += x.mean(dim=0)
    
    weight /= len(train_loader)
    bias /= len(train_loader)

    model.weight.data = (torch.triu(weight, diagonal=1) + torch.triu(weight, diagonal=1).t()) / 2.0
    model.bias.data = bias

def train_iterative(
    model, 
    train_loader,
    optimiser,
    model_name, 
    num_epochs, 
    criterion=None,
    scheduler=None,
    mode='default',
    flatten=False, 
    model_dir="out/weights",
    log_dir="out/logs", 
    step=0, 
    save_model=True,
    untrain_after=None,
    untrain_loss_fn=F.l1_loss,
    untrain_const=0.5,
    eval_loss_every=None,
    eval_recon_loss_every=None,
    device="cpu",
    plot=False,
):
    """
    |  This method implements many different training methods for Hopfield Networks.
    |  hopfield - performs the standard learning procedure for Hopfield Networks iteratively over the dataset.
    |  This is done by taking the outer product of the data with itself and averages over the dataset.
    |  gardner - Improves upon 'hopfield' by only updating the incoming weights of incorrect neurons.
    |  energy - formulates the problem as an energy minimisation problem, and updates the weights to reduce this value.
    |  reconstruction_err - updates the weights to reduce the reconstruction error, which uses mean squared error as its criterion.
    |  PCHNetV2 - aims to combine both reconstruction_err and energy modes, though is not effective in its current state.

    Args:
        |  model (torch.nn.Module): The model to train.
        |  train_loader (torch.utils.data.DataLoader): The training data.
        |  optimiser (torch.optim.Optimizer): The optimiser to use.
        |  model_name (str): The name of the model.
        |  num_epochs (int): The number of epochs to train for.
        |  criterion (torch.nn.Module): The loss function to use.
        |  scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to use.
        |  mode (str): The training mode to use.
        |  flatten (bool): Whether to flatten the input data.
        |  model_dir (str): The directory to save the model weights.
        |  log_dir (str): The directory to save the logs.
        |  step (int): The current step of the training procedure.
        |  save_model (bool): Whether to save the model weights.
        |  untrain_after (int): The number of epochs to train for before untraining the model. If None, the model will not be untrained.
        |  untrain_loss_fn (torch.nn.Module): The loss function to use for untraining.
        |  eval_loss_every (int): The number of epochs to train for before evaluating the loss. If None, the loss will not be evaluated.
        |  eval_recon_loss_every (int): The number of epochs to train for before evaluating the reconstruction loss. If None, the reconstruction loss will not be evaluated.
        |  device (str): The device to use.
        |  plot (bool): Whether to plot the training loss using TensorBoard.

    Returns:
        |  train_loss (list): The training loss for each epoch.
        |  train_energy (list): The training energy for each epoch.
        |  train_recon_loss (list): The training reconstruction loss for each epoch.
        |  step (int): The current step of the training procedure.

    """
    if plot:
        writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_loss = []
    train_recon_loss = []
    train_energy = []
    
    best_train_loss = float("inf")
    assert mode in ['hopfield', 'gardner', 'energy', 'reconstruction_err', 'PCHNetV2', 'pass']
    if save_model:
        assert eval_loss_every is not None, "eval_loss_every must be specified if save_model is True"

    for epoch in range(num_epochs):
        
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        epoch_train_energy = 0.0

        n = 0
        for batch_idx, (images, y) in loop:
            n += images.shape[0]
            if epoch > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_energy = train_energy[-1],
                    train_loss = train_loss[-1], 
                    train_recon_loss = train_recon_loss[-1],
                )

            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)

            optimiser.zero_grad()

            energy = model.calc_energy(x).mean()

            if mode == 'hopfield':
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
            elif mode == 'gardner':
                grad = -torch.bmm(x.unsqueeze(2), x.unsqueeze(1))

                # Determines which neurons are incorrect, and gives them a value of 1, and 0 otherwise.
                next_x = model.step(x, 0)
                is_diff = (next_x != x).float()
                is_diff_mat = is_diff.unsqueeze(1).repeat(1, x.shape[1], 1)

                # Element-wise multiplication to set incoming weights of correct neurons to 0.
                grad = grad * is_diff_mat

                if model.weight.grad is None:
                    model.weight.grad = grad.mean(dim=0)
                else:
                    model.weight.grad += grad.mean(dim=0)
                
                if model.bias is not None:
                    if model.bias.grad is None:
                        model.bias.grad = (x*is_diff).mean(dim=0)
                    else:
                        model.bias.grad += (x*is_diff).mean(dim=0)

            elif mode == 'energy':
                energy.backward()

            elif mode == 'reconstruction_err':
                assert type(model) == PCHNet or type(model) == PCHNetV2, "step_err mode only works with PCHNet"

                # out, e = model.step(x, 0, actv_fn=Tanh(0.1)) # Using Tanh to allow gradients to flow through
                out = model(x) # Using Tanh to allow gradients to flow through
                loss = criterion(out, x)
                loss.backward()
            
            elif mode == "PCHNetV2":
                assert type(model) == PCHNetV2, "PCHNetV2 mode only works with PCHNetV2"
                out, e = model.step(x, 0, actv_fn=Tanh(0.1))
                loss = criterion(out, x)
                loss.backward()
                model.weight.grad.zero_()
                model.bias.grad.zero_()
                energy.backward()

            elif mode == "pass":
                pass
            
            # Untrain to reduce spurious minima
            if untrain_after is not None and epoch > untrain_after:
                untrain_grad(x, model, optimiser, mode, untrain_loss_fn, untrain_const)

            optimiser.step()
            
            with torch.no_grad():
                epoch_train_energy += energy.item()

        train_energy.append(epoch_train_energy / len(train_loader))

        if eval_loss_every is not None:
            if epoch == 0 or (epoch+1) % eval_loss_every == 0:
                with torch.no_grad():
                    model.eval()
                    train_loss.append(evaluate(model, train_loader, flatten=flatten, device=device))
                    if scheduler is not None:
                        scheduler.step(train_loss[-1])
                    if save_model:
                        if best_train_loss > train_loss[-1]:
                            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
                            best_train_loss = train_loss[-1]
                    model.train()

        if eval_recon_loss_every is not None:
            if epoch == 0 or (epoch+1) % eval_recon_loss_every == 0:
                with torch.no_grad():
                    out = model(x)
                    train_recon_loss.append(criterion(out, x))


        step += n
        if plot:
            writer.add_scalar("Training Energy", train_energy[-1], step)
            if eval_loss_every is not None:
                writer.add_scalar("Training Loss", train_loss[-1], step)
            if eval_recon_loss_every is not None:
                writer.add_scalar("Training Reconstruction Loss", train_recon_loss[-1], step)
        
    print(f"Best train loss: {best_train_loss}")
    return torch.tensor(train_energy), torch.tensor(train_loss), torch.tensor(train_recon_loss), step
