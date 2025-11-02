# trainers.py

import torch
from tqdm import tqdm
import utils
import configs

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one full training epoch.
    """
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    
    # Using tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        accuracy = utils.calculate_accuracy(logits, labels)
        total_accuracy += accuracy * inputs.size(0)
        
        pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_acc


# trainers.py (modify the evaluate function)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the given dataset.
    """
    model.eval()
    running_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # --- THIS IS THE FIX ---
            # If the model returns a tuple (logits, hint), only use the logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            accuracy = utils.calculate_accuracy(logits, labels)
            total_accuracy += accuracy * inputs.size(0)
            
            pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_acc