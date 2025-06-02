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

def _format_probabilities(logits_tensor: torch.Tensor) -> list[list[float]]:
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits_tensor, dim=1)
    # Move to CPU and convert to a list of lists
    probabilities_list = probabilities.cpu().tolist()
    
    formatted_probabilities_batch = []
    for sample_probs in probabilities_list:
        # Format each probability in the sample to .4f
        formatted_sample = [float(f"{p:.4f}") for p in sample_probs]
        formatted_probabilities_batch.append(formatted_sample)
    return formatted_probabilities_batch

def predict_logits(data_loader: DataLoader, model: nn.Module, device: torch.device, task: str = ''): 
    model.eval()  # Set the model to evaluation mode
    results = []  # List to store prediction results
    num_batches = len(data_loader)  # Total number of batches
    
    with torch.no_grad():  # Disable gradient calculations for inference
        for batch_idx, (inputs, sw_mode) in enumerate(data_loader):
            # Move inputs and switch mode tensor to the specified device (e.g., GPU or CPU)
            inputs, sw_mode = inputs.to(device), sw_mode.to(device)
            
            if task == 'all':
                # Forward pass: model returns logits for gender, hand, year, and level
                # Assumes model output order is: gender, hand, year, level
                gender_logits, hand_logits, year_logits, level_logits = model(inputs, sw_mode)
                
                # Convert logits to formatted probabilities for each task
                formatted_gender_probs = _format_probabilities(gender_logits)
                formatted_hand_probs = _format_probabilities(hand_logits)
                formatted_year_probs = _format_probabilities(year_logits)
                formatted_level_probs = _format_probabilities(level_logits)
                
                # Aggregate results for the current batch
                # Each item in current_batch_results will be a dictionary for one sample
                current_batch_results = []
                # Iterate over each sample in the batch (inputs.size(0) is the batch size)
                for i in range(inputs.size(0)): 
                    sample_data = {
                        'gender_probs': formatted_gender_probs[i],
                        'hand_probs': formatted_hand_probs[i],
                        'year_probs': formatted_year_probs[i],
                        'level_probs': formatted_level_probs[i]
                    }
                    current_batch_results.append(sample_data)
                
                # Extend the main results list with the results from the current batch
                results.extend(current_batch_results)
                
                # Print predictions for the first sample in the batch for all tasks
                if current_batch_results:  # Check if the batch was not empty
                    first_sample_preds = current_batch_results[0]
                    print_msg = (
                        f"Testing Batch: [{batch_idx + 1}/{num_batches}], "
                        f"Prediction (Sample 0 for 'all' tasks):\n"
                        f"  Gender Probs: {first_sample_preds['gender_probs']}\n"
                        f"  Hand Probs:   {first_sample_preds['hand_probs']}\n"
                        f"  Year Probs:   {first_sample_preds['year_probs']}\n"
                        f"  Level Probs:  {first_sample_preds['level_probs']}"
                    )
                    print(print_msg)
                    
            else: # Handle single task prediction
                # Forward pass: model returns logits for a single task
                logits = model(inputs, sw_mode)
                
                # Convert logits to formatted probabilities
                # formatted_probs_batch is a list of lists (batch_size x num_classes)
                formatted_probs_batch = _format_probabilities(logits)
                
                # Extend the main results list
                results.extend(formatted_probs_batch)
                
                # Print prediction for the first sample in the batch
                if formatted_probs_batch: # Check if the batch was not empty
                    # The original print statement for the 'else' case:
                    print(f"Testing Batch: [{batch_idx + 1}/{num_batches}], Prediction: {formatted_probs_batch[0]}")
                    
    return results

def predict_val(data_loader: DataLoader, model: nn.Module, device: torch.device, task: str = ''):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []  # List to store prediction results
    num_batches = len(data_loader)  # Total number of batches
    
    with torch.no_grad():  # Disable gradient calculations for inference
        for batch_idx, (inputs, sw_mode, label) in enumerate(data_loader):
            if task == 'gender':
                label = label[:, 0]
            elif task == 'hand':
                label = label[:, 1]
            elif task == 'year':
                label = label[:, 2]
            elif task == 'level':
                label = label[:, 3]
            else:
                print(f"Validation: Task {task} not supported")
            
            # Move inputs and switch mode tensor to the specified device (e.g., GPU or CPU)
            inputs, sw_mode = inputs.to(device), sw_mode.to(device)
            logits = model(inputs, sw_mode)
            
            if task == 'gender' or task =='hand':
                pred = torch.argmax(logits, dim=1)
                y_true.extend(label.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                print(f"Testing Batch: [{batch_idx + 1}/{num_batches}], Prediction: {pred}")
            else:
                pred = _format_probabilities(logits)
                y_true.extend(label.cpu().numpy())
                y_pred.extend([pred[0]])
                print(f"Testing Batch: [{batch_idx + 1}/{num_batches}], Prediction: {[pred[0]]}")
                
        return y_true, y_pred
                
            
            

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