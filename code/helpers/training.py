import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os

def train_epoch(data_loader: DataLoader, model: nn.Module, loss_fn, optimizer: optim.Optimizer, device: torch.device, task = ''):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    num_batches = len(data_loader) # Get total number of batches
    for batch_idx, (inputs, _, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if task == 'gender':
            targets = targets[:, 0]
        elif task == 'hand':
            targets = targets[:, 1]
        elif task == 'year':
            targets = targets[:, 2]
        elif task == 'level':
            targets = targets[:, 3]
        else:
            print(f"Training: Task {task} not supported")
            exit()

        targets = targets.long()
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs) # Forward pass
        # print("model out:", outputs, targets)
        loss = loss_fn(outputs, targets) # Calculate loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Print current batch / total batch
        print(f"  Training Batch: [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.4f}")

    if num_batches == 0:
        return 0.0
    avg_epoch_loss = running_loss / num_batches
    return avg_epoch_loss

def val_epoch(data_loader: DataLoader, model: nn.Module, loss_fn, device: torch.device):
    model.eval()
    running_loss = 0.0
    num_batches = len(data_loader) # Get total number of batches
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs) # Forward pass
            loss = loss_fn(outputs, targets) # Calculate loss

            running_loss += loss.item()

            # Print current batch / total batch
            print(f"  Validation Batch: [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.4f}")

    if num_batches == 0:
        return 0.0
    avg_epoch_loss = running_loss / num_batches
    return avg_epoch_loss

def train_val_model(train_loader: DataLoader,
                    val_loader: DataLoader,
                    model: nn.Module,
                    epochs: int,
                    loss_fn,
                    optimizer: optim.Optimizer,
                    device: torch.device = None,
                    task = ''):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu" # Original commented line retained
    print(f"Using device: {device}")
    model.to(device)

    save_dir = "trained_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses_per_epoch = []
    val_losses_per_epoch = []
    best_val_loss = float('inf')
    previous_best_model_path = None

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        # Training phase
        print("Training Phase...")
        avg_train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, task)
        train_losses_per_epoch.append(avg_train_loss)

        # Validation phase
        print("Validation Phase...")
        avg_val_loss = val_epoch(val_loader, model, loss_fn, device)
        val_losses_per_epoch.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Summary | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Remove the previously saved best model, if any
            if previous_best_model_path and os.path.exists(previous_best_model_path):
                try:
                    os.remove(previous_best_model_path)
                except OSError as e:
                    print(f"Error removing old best model {previous_best_model_path}: {e}")

            # Save the new best model
            current_epoch_for_filename = epoch + 1 # 1-indexed epoch
            model_filename = f"{task}_best_model_epx{current_epoch_for_filename}.pth"
            current_best_model_path = os.path.join(save_dir, model_filename)
            torch.save(model.state_dict(), current_best_model_path)
            previous_best_model_path = current_best_model_path # Update path for next potential deletion
            print(f"Saved new best model to {current_best_model_path} (Val Loss: {best_val_loss:.4f})")
        
        print("-" * 30) # Separator for epochs

    print(f"Training finished. Final best model saved at: {previous_best_model_path if previous_best_model_path else 'N/A'}")
    return train_losses_per_epoch, val_losses_per_epoch