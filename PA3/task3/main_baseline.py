# main_baseline_hub.py

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
import utils

def main():
    """Main function to run the NEW, FAIR baseline training using the torch.hub student."""
    
    # For reproducibility
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    # --- Setup Logging ---
    experiment_name = "student_hub_baseline"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
    csv_path = os.path.join("results", f"{experiment_name}_log.csv")
    os.makedirs("results", exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])
    print(f"TensorBoard logs will be saved to: {os.path.join('logs', experiment_name)}")
    print(f"Results CSV will be saved to: {csv_path}")


    # --- 1. Load Data ---
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = data_loader.get_cifar100_loaders()
    print("Dataset loaded successfully.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    print("Initializing torch.hub VGG-11 student model...")
    
    # --- THIS IS THE KEY CHANGE ---
    # Load the CIFAR-specialized VGG-11 architecture from the hub, but NOT pretrained.
    student_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_vgg11_bn",
        pretrained=False  # Train from scratch
    ).to(configs.DEVICE)
    
    # Standard Cross-Entropy loss for baseline
    criterion = nn.CrossEntropyLoss()
    
    # Use the original, higher learning rate for training from scratch
    optimizer = optim.SGD(
        student_model.parameters(), 
        lr=configs.LEARNING_RATE, # Using 0.1
        momentum=configs.MOMENTUM,
        weight_decay=configs.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=configs.LR_SCHEDULER_STEP_SIZE, 
        gamma=configs.LR_SCHEDULER_GAMMA
    )
    
    print("Model and optimizer initialized.")
    print(f"--- Starting Training: {experiment_name} ---")
    print(f"Training on {configs.DEVICE}...")

    # --- 3. Training Loop ---
    # This loop is identical to our very first baseline script
    best_acc = 0.0
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        
        train_loss, train_acc = trainers.train_epoch(
            student_model, train_loader, criterion, optimizer, configs.DEVICE
        )
        test_loss, test_acc = trainers.evaluate(
            student_model, test_loader, criterion, configs.DEVICE
        )
        
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        csv_writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            utils.save_checkpoint(
                student_model, optimizer, epoch, f"{experiment_name}_best.pth.tar"
            )
            
    print(f"\n--- Training Finished: {experiment_name} ---")
    print(f"Best Test Accuracy for Hub Student Baseline: {best_acc:.2f}%")
    
    writer.close()
    csv_file.close()

if __name__ == '__main__':
    main()