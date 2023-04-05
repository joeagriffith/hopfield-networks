import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from tqdm import tqdm
from utils.extra_funcs import RandomGaussianNoise
from utils.eval import evaluate_denoise
    
def train_denoise(
    model, 
    train_dataset,
    val_dataset,
    optimiser, 
    criterion, 
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    save_model=True,
    batch_size=100,
    minimise='loss',
    learning_rate=3e-4,
    weight_decay=1e-2,
    device="cpu",
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    # train_loss = []
    train_energy = []
    # val_loss = []
    # val_energy = []

    assert minimise in ['loss', 'energy'], "minimise must be either 'loss' or 'energy'"
    
    #  For determining best model
    # best_val_loss = float("inf")

    scheduler = ReduceLROnPlateau(optimiser, mode='min', min_lr=3e-9, factor=0.1, patience=0, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    noiser = RandomGaussianNoise(mean=0.0, std=0.1)

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        # epoch_train_loss = 0.0
        epoch_train_energy = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            # target = x.clone()

            # Add noise
            # x = noiser(x)

            # train for each step
            # for _ in range(model.steps):
            #     x = model.step(x)
            #     loss = criterion(x, target)
            #     optimiser.zero_grad()
            #     if minimise == 'loss':
            #         loss.backward()
            #     elif minimise == 'energy':
            #         energy = model.calc_energy(x).mean()
            #         energy.backward()
            #     optimiser.step()

            #     x = x.detach()

            optimiser.zero_grad()
            energy = model.calc_energy(x).mean()
            energy.backward()
            optimiser.step()
            

            with torch.no_grad():
                # epoch_train_loss += loss.item()
                # epoch_train_energy += model.calc_energy(x).mean().item()
                epoch_train_energy += energy.item()

                if epoch > 0:
                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(
                        train_energy = train_energy[-1],
                        # train_loss = train_loss[-1], 
                        # val_loss = val_loss[-1], 
                    )

        # train_loss.append(epoch_train_loss / len(train_loader))
        # print(train_loss)
        train_energy.append(epoch_train_energy / len(train_loader))

        scheduler.step(energy)
        
        # epoch_val_loss, epoch_val_energy = evaluate_denoise(model, val_loader, criterion, device, flatten)
        # val_loss.append(epoch_val_loss)
        # val_energy.append(epoch_val_energy)
            
        if save_model:
            # if best_val_loss > val_loss[-1]:
                # best_val_loss = val_loss[-1]
                torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')

        step += len(train_dataset)
        # writer.add_scalar("Training Loss", train_loss[-1], step)
        writer.add_scalar("Training Energy", train_energy[-1], step)
        # writer.add_scalar("Validation Loss", val_loss[-1], step)
        # writer.add_scalar("Validation Energy", val_energy[-1], step)
        
    # return torch.tensor(train_loss), torch.tensor(train_energy), torch.tensor(val_loss), torch.tensor(val_energy), step
    return torch.tensor(train_energy), step
