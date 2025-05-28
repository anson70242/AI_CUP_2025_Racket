import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def predict(data_loader: DataLoader, model: nn.Module, device: torch.device, task = ''):
    model.eval()
    results = []
    num_batches = len(data_loader)
    with torch.no_grad():
        for batch_idx, (inputs, sw_mode) in enumerate(data_loader):
            inputs, sw_mode = inputs.to(device), sw_mode.to(device) 
            logits = model(inputs, sw_mode) # Forward pass
            predictions = torch.argmax(logits, dim=1)
            results.extend(predictions.cpu().tolist())
            print(f"Testing Batch: [{batch_idx + 1}/{num_batches}], Prediction: {predictions.item()}")
    return results

def predict_logits(data_loader: DataLoader, model: nn.Module, device: torch.device, task = ''): 
    model.eval() 
    results = [] 
    num_batches = len(data_loader) 
    with torch.no_grad(): 
        for batch_idx, (inputs, sw_mode) in enumerate(data_loader): 
            inputs, sw_mode = inputs.to(device), sw_mode.to(device) 
            logits = model(inputs, sw_mode) # Forward pass 
            probs = torch.softmax(logits, dim=1).cpu().tolist()

            # convert all element in probs to .4f
            formatted_probs_batch = []
            for prob in probs:
                formatted_prob = [float(f"{p:.4f}") for p in prob]
                formatted_probs_batch.append(formatted_prob)

            results.extend(formatted_probs_batch)
            print(f"Testing Batch: [{batch_idx + 1}/{num_batches}], Prediction: {formatted_probs_batch[0]}")
    return results

def plot_loss(train_losses, val_losses, task='none'):
    """
    Plots the training and validation losses.

    Args:
        train_losses (list or array-like): A list of training loss values.
        val_losses (list or array-like): A list of validation loss values.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability

    # Plot training loss
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')  # 'bo-' for blue circle markers and solid line

    # Plot validation loss
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')  # 'ro-' for red circle markers and solid line

    # Add title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Add a grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.savefig(f"{task}_plot_test.png")

# if __name__ == '__main__':