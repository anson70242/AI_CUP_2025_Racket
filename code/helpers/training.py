import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os

def train_epoch(data_loader: DataLoader, model: nn.Module, loss_fn, optimizer: optim.Optimizer, device: torch.device, task = ''):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    num_batches = len(data_loader) # Get total number of batches
    for batch_idx, (inputs, sw_mode, targets) in enumerate(data_loader):
        inputs, targets, sw_mode = inputs.to(device), targets.to(device), sw_mode.to(device)
        if task == 'gender':
            targets = targets[:, 0]
        elif task == 'hand':
            targets = targets[:, 1]
        elif task == 'year':
            targets = targets[:, 2]
        elif task == 'level':
            targets = targets[:, 3]
        elif task == 'all':
            targets = targets
        else:
            print(f"Training: Task {task} not supported")
            exit()

        targets = targets.long()
        
        optimizer.zero_grad()  # Clear previous gradients
        
        # print("model out:", outputs, targets)
        if task == 'all':
            outputs_gender, outputs_hand, outputs_year, outputs_level = model(inputs, sw_mode) # Forward pass
            # Calculate loss
            gender_loss_raw = loss_fn(outputs_gender, targets[:, 0])
            hand_loss_raw = loss_fn(outputs_hand, targets[:, 1])
            year_loss_raw = loss_fn(outputs_year, targets[:, 2])
            level_loss_raw = loss_fn(outputs_level, targets[:, 3])
            
            loss_gender = 0.5 * torch.exp(-model.log_var_gender) * gender_loss_raw + 0.5 * model.log_var_gender
            loss_hand   = 0.5 * torch.exp(-model.log_var_hand) * hand_loss_raw   + 0.5 * model.log_var_hand
            loss_year   = 0.5 * torch.exp(-model.log_var_year) * year_loss_raw   + 0.5 * model.log_var_year
            loss_level  = 0.5 * torch.exp(-model.log_var_level) * level_loss_raw  + 0.5 * model.log_var_level
            loss = loss_gender + loss_hand + loss_year + loss_level
        else:
            outputs = model(inputs, sw_mode) # Forward pass
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

def val_epoch(data_loader: DataLoader, model: nn.Module, loss_fn, device: torch.device, task = ''):
    model.eval()
    running_loss = 0.0
    num_batches = len(data_loader) # Get total number of batches
    with torch.no_grad():
        for batch_idx, (inputs, sw_mode, targets) in enumerate(data_loader):
            inputs, targets_all, sw_mode = inputs.to(device), targets.to(device), sw_mode.to(device) # Renamed targets to targets_all

            if task == 'gender':
                current_targets = targets_all[:, 0]
            elif task == 'hand':
                current_targets = targets_all[:, 1]
            elif task == 'year':
                current_targets = targets_all[:, 2]
            elif task == 'level':
                current_targets = targets_all[:, 3]
            else:
                print(f"Validation: Task {task} not supported")

            current_targets = current_targets.long()

            outputs = model(inputs, sw_mode) # Forward pass
            loss = loss_fn(outputs, current_targets) # Calculate loss

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
                    task = '',
                    early_stop_patience=20):
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
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        # Training phase
        print("Training Phase...")
        avg_train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device, task)
        train_losses_per_epoch.append(avg_train_loss)

        # Validation phase
        print("Validation Phase...")
        avg_val_loss = val_epoch(val_loader, model, loss_fn, device, task)
        val_losses_per_epoch.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Summary | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
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
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
        
        print("-" * 30) # Separator for epochs
        
        if epochs_no_improve >= early_stop_patience:
            print(f"✋ Early stopping triggered after {epoch + 1} epochs "
                  f"due to no improvement in validation loss for {early_stop_patience} consecutive epochs.")
            break # Exit the training loop

    print(f"Training finished. Final best model saved at: {previous_best_model_path if previous_best_model_path else 'N/A'}")
    if not previous_best_model_path and epochs > 0:
        print("⚠️ No model was saved as best model. This might happen if validation loss never improved or if epochs_no_improve was always high.")
    elif epochs == 0:
        print("⚠️ Training was set for 0 epochs.")
        
    return train_losses_per_epoch, val_losses_per_epoch