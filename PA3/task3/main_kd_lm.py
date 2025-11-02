# main_kd_lm.py (Updated for Fair Comparison)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import csv
from torch.utils.tensorboard import SummaryWriter

# Import our modular components
import configs
import data_loader
import trainers
import trainers_kd
import utils
import losses

def main():
    """Main function to run the FAIR Logit Matching KD experiment using Hub models."""
    
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    # --- Logging ---
    # Use a new name to distinguish this fair run in the logs
    experiment_name = "student_kd_lm_hub_fair"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
    csv_path = os.path.join("results", f"{experiment_name}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    # --- 1. Load Data ---
    train_loader, test_loader = data_loader.get_cifar100_loaders()

    # --- 2. Initialize Models (The FAIR way) ---
    print("Loading Hub student and teacher models...")
    
    # Student model (Hub VGG-11, trained from scratch)
    student_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_vgg11_bn",
        pretrained=False
    ).to(configs.DEVICE)
    
    # Teacher model (Hub VGG-16, pretrained)
    teacher_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_vgg16_bn",
        pretrained=True
    ).to(configs.DEVICE)
    teacher_model.eval()

    print("Models loaded successfully.")

    # --- 3. Initialize Loss, Optimizer ---
    # Use our standard DistillationLoss with the tuned alpha
    criterion_kd = losses.DistillationLoss(alpha=0.5, temperature=configs.KD_TEMPERATURE)
    criterion_eval = nn.CrossEntropyLoss()
    
    # Use the safer, smaller learning rate for KD
    optimizer = optim.SGD(student_model.parameters(), lr=configs.KD_LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    print(f"--- Starting Training: {experiment_name} ---")

    # --- 4. Training Loop ---
    best_acc = 0.0
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        
        # Use the generic KD trainer, which now works perfectly
        train_loss, train_acc = trainers_kd.train_epoch_kd(
            student_model, teacher_model, train_loader, criterion_kd, optimizer, configs.DEVICE
        )
        # The generic evaluate function also works perfectly
        test_loss, test_acc = trainers.evaluate(
            student_model, test_loader, criterion_eval, configs.DEVICE
        )
        
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        csv_writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])

        if test_acc > best_acc:
            best_acc = test_acc
            utils.save_checkpoint(student_model, optimizer, epoch, f"{experiment_name}_best.pth.tar")
            
    print(f"\n--- Training Finished: {experiment_name} ---")
    print(f"Best Test Accuracy for Distilled Student (LM, Fair): {best_acc:.2f}%")
    
    writer.close()
    csv_file.close()

if __name__ == '__main__':
    main()