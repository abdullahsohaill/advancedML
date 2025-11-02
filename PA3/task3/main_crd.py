# main_crd.py (Updated for Fair Comparison)

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
from models_with_embeddings import VGGWithEmbedding # Reusing our wrapper
import trainers
import trainers_kd
import utils
import losses

def main():
    """Main function to run the FAIR CRD experiment using Hub models."""
    
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    experiment_name = "crd_new"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))
    csv_path = os.path.join("results", f"{experiment_name}_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    train_loader, test_loader = data_loader.get_cifar100_loaders()

    print("Loading and wrapping Hub models for CRD...")

    # 1. Load the base models from torch.hub
    student_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    
    # 2. Wrap the base models to get embeddings
    student_model = VGGWithEmbedding(base_model=student_base).to(configs.DEVICE)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    teacher_model.eval()
    
    # Let's confirm the embedding sizes match
    dummy_input = torch.randn(2, 3, 32, 32).to(configs.DEVICE)
    _, s_emb = student_model(dummy_input)
    _, t_emb = teacher_model(dummy_input)
    print(f"Student Embedding Shape: {s_emb.shape}") # Should be [2, 512]
    print(f"Teacher Embedding Shape: {t_emb.shape}") # Should be [2, 512]
    assert s_emb.shape == t_emb.shape, "Embedding dimensions must match!"

    print("Models loaded and wrapped successfully. Embedding dimensions match.")

    # 3. Define Loss functions and Optimizer
    criterion_kd = losses.CRDLoss(
        alpha=0.5, 
        kd_temp=configs.KD_TEMPERATURE, # The high temp (4.0) for logits
        crd_temp=configs.CRD_CONTRASTIVE_TEMP, # The low temp (0.1) for contrastive
        lambda_crd=configs.CRD_LAMBDA 
    )
    criterion_eval = nn.CrossEntropyLoss()
    
    # Use the proven high learning rate
    optimizer = optim.SGD(student_model.parameters(), lr=configs.LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    print(f"--- Starting Training: {experiment_name} ---")

    best_acc = 0.0
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        
        # Use the CRD-specific trainer
        train_loss, train_acc = trainers_kd.train_epoch_crd(
            student_model, teacher_model, train_loader, criterion_kd, optimizer, configs.DEVICE
        )
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
    print(f"Best Test Accuracy for Distilled Student (CRD, Fair): {best_acc:.2f}%")
    
    writer.close()
    csv_file.close()

if __name__ == '__main__':
    main()