# main_kd_lm_vgg19.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import csv
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Import our modular components
import configs
import data_loader
import trainers_kd
import trainers # For the evaluate function
import utils
import losses

def main():
    """Main function to test the efficacy of a larger VGG-19 teacher using Logit Matching."""
    
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    # --- Logging ---
    experiment_name = "student_lm_with_vgg19_teacher"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
    csv_path = os.path.join("results", f"{experiment_name}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    # --- 1. Load Data ---
    train_loader, test_loader = data_loader.get_cifar100_loaders()

    # --- 2. Initialize Models ---
    print("Loading Hub student (VGG-11) and LARGE teacher (VGG-19)...")
    
    # Student model (VGG-11, trained from scratch)
    student_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_vgg11_bn",
        pretrained=False
    ).to(configs.DEVICE)
    
    # --- THIS IS THE KEY CHANGE ---
    # Teacher model is now the larger VGG-19, pretrained on CIFAR-100
    teacher_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_vgg19_bn", # <-- Changed from vgg16
        pretrained=True
    ).to(configs.DEVICE)
    teacher_model.eval()
    # --- END OF CHANGE ---

    print("Models loaded successfully.")

    # --- 3. Initialize Loss, Optimizer ---
    criterion_kd = losses.DistillationLoss(alpha=0.5, temperature=configs.KD_TEMPERATURE)
    criterion_eval = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(student_model.parameters(), lr=configs.KD_LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    print(f"--- Starting Training: {experiment_name} ---")

    # --- 4. Training Loop ---
    best_acc = 0.0
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        
        # We need to handle the tuple output from our wrapped models
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        total_accuracy = 0.0
        
        pbar = tqdm(train_loader, desc="Training (KD)", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(configs.DEVICE), labels.to(configs.DEVICE)
            optimizer.zero_grad()
            with torch.no_grad(): teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            loss = criterion_kd(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            s_logits = student_outputs[0] if isinstance(student_outputs, tuple) else student_outputs
            accuracy = utils.calculate_accuracy(s_logits, labels)
            total_accuracy += accuracy * inputs.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = total_accuracy / len(train_loader.dataset)
        
        test_loss, test_acc = trainers.evaluate(
            student_model, test_loader, criterion_eval, configs.DEVICE
        )
        
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        csv_writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])

        if test_acc > best_acc:
            best_acc = test_acc
            utils.save_checkpoint(student_model, optimizer, epoch, f"{experiment_name}_best.pth.tar")
            
    print(f"\n--- Training Finished: {experiment_name} ---")
    print(f"Best Test Accuracy for Student (trained by VGG-19): {best_acc:.2f}%")
    
    writer.close()
    csv_file.close()

if __name__ == '__main__':
    main()