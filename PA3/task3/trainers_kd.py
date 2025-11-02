# trainers_kd.py (Updated to handle tuple outputs for accuracy calculation)

import torch
from tqdm import tqdm
import utils

def train_epoch_kd(student_model, teacher_model, dataloader, criterion, optimizer, device):
    """
    Performs one full training epoch for Knowledge Distillation (LM, DKD).
    """
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(dataloader, desc="Training (KD)", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        student_outputs = student_model(inputs)
        
        loss = criterion(student_outputs, teacher_outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # --- THIS IS THE FIX ---
        # Unpack the logits from the student's output before calculating accuracy
        s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
        accuracy = utils.calculate_accuracy(s_logits, labels)
        # --- END OF FIX ---
        
        total_accuracy += accuracy * inputs.size(0)
        
        pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def train_epoch_crd(student_model, teacher_model, dataloader, criterion, optimizer, device):
    """
    Performs one full training epoch for Contrastive Representation Distillation.
    """
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(dataloader, desc="Training (CRD)", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits, teacher_embedding = teacher_model(inputs)

        student_logits, student_embedding = student_model(inputs)
        
        # --- THIS IS THE FIX ---
        # Revert the call to pass five separate arguments, as the function expects.
        loss = criterion(student_logits, teacher_logits, student_embedding, teacher_embedding, labels)
        # --- END OF FIX ---

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy is calculated ONLY on the logits (this part is correct)
        accuracy = utils.calculate_accuracy(student_logits, labels)
        total_accuracy += accuracy * inputs.size(0)
        
        pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_acc

# The train_epoch_hint function also needs this fix if you use it, but it's not
# part of the color invariance experiment. For completeness, I'll update it too.
def train_epoch_hint(student_model, teacher_model, dataloader, criterion, optimizer, device):
    """
    Performs one full training epoch for Hint-based Knowledge Distillation.
    """
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(dataloader, desc="Training (Hint)", unit="batch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits, teacher_hint = teacher_model(inputs)

        student_logits, student_hint = student_model(inputs)
        
        loss = criterion(student_logits, teacher_logits, student_hint, teacher_hint, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # --- THIS IS THE FIX (also needed here) ---
        accuracy = utils.calculate_accuracy(student_logits, labels)
        # --- END OF FIX ---
        
        total_accuracy += accuracy * inputs.size(0)
        
        pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_acc