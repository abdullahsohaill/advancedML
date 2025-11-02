# utils.py

import torch
import os
import configs

def save_checkpoint(model, optimizer, epoch, filename):
    """Saves model checkpoint."""
    if not os.path.exists(configs.CHECKPOINT_DIR):
        os.makedirs(configs.CHECKPOINT_DIR)
        
    filepath = os.path.join(configs.CHECKPOINT_DIR, filename)
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)
    print(f"==> Saved checkpoint to {filepath}")

def calculate_accuracy(outputs, targets):
    """Calculates top-1 accuracy."""
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
    return (correct_k.mul_(100.0 / batch_size)).item()