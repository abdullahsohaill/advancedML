# main_fitnets_final_stage2.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import configs
import data_loader
from models_with_hints import VGGWithHint # Still need the wrapper to load the state_dict
import trainers
import trainers_kd # Using our standard KD trainer
import utils
import losses

def main():
    """STAGE 2 (FINAL): Trains the ENTIRE student using Stage 1 weights as initialization."""
    
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    experiment_name = "student_fitnets_final_stage2"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
    csv_path = os.path.join("results", f"{experiment_name}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    train_loader, test_loader = data_loader.get_cifar100_loaders()

    print("Loading and preparing models for Stage 2...")

    # We need to build the model with the wrapper to load the state_dict correctly
    student_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    student_model = VGGWithHint(base_model=student_base, hint_layer_index=8).to(configs.DEVICE)

    # The teacher doesn't need to be a hint model anymore, just the base model
    teacher_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True).to(configs.DEVICE)
    teacher_model.eval()
    
    # --- LOAD PRE-TRAINED WEIGHTS FROM STAGE 1 ---
    checkpoint_path = os.path.join(configs.CHECKPOINT_DIR, "student_fitnets_final_pretrained.pth.tar")
    print(f"Loading pre-trained student from {checkpoint_path}")
    student_model.load_state_dict(torch.load(checkpoint_path))

    # --- NO FREEZING ---
    
    # Define loss and optimizer
    # Use our best, stable DistillationLoss
    criterion_kd = losses.DistillationLoss(configs.KD_ALPHA, temperature=configs.KD_TEMPERATURE)
    criterion_eval = nn.CrossEntropyLoss()
    
    # Optimizer trains ALL parameters of the student model
    optimizer = optim.SGD(student_model.parameters(), lr=configs.LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    print(f"--- Starting Training: {experiment_name} ---")
    
    best_acc = 0.0
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        
        # We need a custom loop because our student model still returns a tuple
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        total_accuracy = 0.0
        
        pbar = tqdm(train_loader, desc=f"Stage 2, Epoch {epoch+1}/{configs.EPOCHS}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(configs.DEVICE), labels.to(configs.DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # Unpack the tuple from the student model
            student_logits, _ = student_model(inputs)

            # Pass only the logits to the loss function
            loss = criterion_kd(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            accuracy = utils.calculate_accuracy(student_logits, labels)
            total_accuracy += accuracy * inputs.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy:.2f}%")

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = total_accuracy / len(train_loader.dataset)
        
        test_loss, test_acc = trainers.evaluate(student_model, test_loader, criterion_eval, configs.DEVICE)
        
        scheduler.step()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        csv_writer.writerow([epoch + 1, train_loss, train_acc, test_loss, test_acc])
        
        if test_acc > best_acc:
            best_acc = test_acc
            utils.save_checkpoint(student_model, optimizer, epoch, f"{experiment_name}_best.pth.tar")

    print(f"\n--- Training Finished: {experiment_name} ---")
    print(f"Best Test Accuracy for Distilled Student (FitNets, 2-Stage Final): {best_acc:.2f}%")
    
    writer.close()
    csv_file.close()

if __name__ == '__main__':
    main()