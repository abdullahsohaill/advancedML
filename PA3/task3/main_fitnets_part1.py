# main_fitnets_final_stage1.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import configs
import data_loader
from models_with_hints import VGGWithHint
import utils

def main():
    """STAGE 1 (FINAL): Pre-trains the Hub student's early layers using only the hint loss."""
    
    torch.manual_seed(configs.RANDOM_SEED)
    random.seed(configs.RANDOM_SEED)
    np.random.seed(configs.RANDOM_SEED)
    
    experiment_name = "student_fitnets_final_stage1"
    writer = SummaryWriter(log_dir=os.path.join("logs", experiment_name))

    train_loader, _ = data_loader.get_cifar100_loaders()

    print("Loading and preparing Hub models for Stage 1...")

    student_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)

    student_model = VGGWithHint(base_model=student_base, hint_layer_index=8).to(configs.DEVICE)
    teacher_model = VGGWithHint(base_model=teacher_base, hint_layer_index=16).to(configs.DEVICE)
    teacher_model.eval()
    
    # --- FREEZING LOGIC ---
    # Freeze the student's later layers (features part 2 and classifier)
    # We only want to train the first 9 layers of the features module
    for param in student_model.features[9:].parameters():
        param.requires_grad = False
    for param in student_model.classifier.parameters():
        param.requires_grad = False

    # Define loss and optimizer
    criterion_hint = nn.MSELoss()
    
    # Optimizer trains ONLY the early layers of the student (the first 9 layers in .features)
    params_to_train = student_model.features[:9].parameters()
    optimizer = optim.SGD(params_to_train, lr=configs.KD_LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    
    print(f"--- Starting Training: {experiment_name} ---")

    for epoch in range(configs.HINT_STAGE1_EPOCHS):
        student_model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Stage 1, Epoch {epoch+1}/{configs.HINT_STAGE1_EPOCHS}", unit="batch")
        for inputs, _ in pbar:
            inputs = inputs.to(configs.DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_hint = teacher_model(inputs)

            _, student_hint = student_model(inputs)

            loss = criterion_hint(student_hint, teacher_hint)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(mse_loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/hint_pretrain', epoch_loss, epoch)
        print(f"Epoch {epoch+1} MSE Loss: {epoch_loss:.6f}")

    # --- SAVE THE PRE-TRAINED MODEL ---
    save_path = os.path.join(configs.CHECKPOINT_DIR, "student_fitnets_final_pretrained.pth.tar")
    torch.save(student_model.state_dict(), save_path)
    
    print(f"\n--- Stage 1 Finished ---")
    print(f"Pre-trained student saved to {save_path}")
    
    writer.close()

if __name__ == '__main__':
    main()