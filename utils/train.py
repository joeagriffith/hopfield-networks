import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
from utils.eval import evaluate_noise, evaluate_mask
    
def train_denoise(
    model, 
    train_dataset,
    optimiser,
    scheduler,
    model_name, 
    num_epochs, 
    flatten=False, 
    model_dir="models",
    log_dir="logs", 
    step=0, 
    save_model=True,
    batch_size=100,
    validate_every=None,
    device="cpu",
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")
    train_loss = []
    train_energy = []
    
    best_train_loss = float("inf")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        
        model.train()
        train_dataset.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        epoch_train_energy = 0.0

        for batch_idx, (images, y) in loop:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)

            optimiser.zero_grad()
            energy = model.calc_energy(x).mean()
            energy.backward()
            optimiser.step()
            

            with torch.no_grad():
                epoch_train_energy += energy.item()

                if epoch > 0:
                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(
                        train_energy = train_energy[-1],
                        train_loss = train_loss[-1], 
                    )

        train_energy.append(epoch_train_energy / len(train_loader))

        if validate_every is not None and epoch % validate_every == 0:
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
