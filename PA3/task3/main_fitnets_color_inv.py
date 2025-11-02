# main_fitnets_color_inv.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

import configs
import data_loader
from models_with_hints import VGGWithHint
import trainers
import utils
import losses

def main():
    """Trains a student using FitNets, distilled from the color-invariant teacher."""
    
    experiment_name = "student_fitnets_from_color_inv_teacher"
    
    # --- Load Data (Regular augmentations for student) ---
    train_loader, test_loader = data_loader.get_cifar100_loaders()

    # --- Load Models ---
    print("--- Preparing for FitNets Stage 1 (Hint Training) ---")
    student_base_s1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    student_s1 = VGGWithHint(base_model=student_base_s1, hint_layer_index=8).to(configs.DEVICE)
    
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=False)
    teacher_model = VGGWithHint(base_model=teacher_base, hint_layer_index=16).to(configs.DEVICE)
    teacher_path = os.path.join(configs.CHECKPOINT_DIR, "teacher_color_invariant.pth.tar")
    # Note: We need VGGWithHint for the teacher here to get the hint layer
    teacher_model.load_state_dict(torch.load(teacher_path))
    teacher_model.eval()

    # --- STAGE 1: Train with Hint Loss ---
    for param in student_s1.features[9:].parameters(): param.requires_grad = False
    for param in student_s1.classifier.parameters(): param.requires_grad = False
    
    criterion_hint = nn.MSELoss()
    optimizer_s1 = optim.SGD(student_s1.features[:9].parameters(), lr=configs.KD_LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    
    for epoch in range(configs.HINT_STAGE1_EPOCHS):
        student_s1.train()
        pbar = tqdm(train_loader, desc=f"Stage 1, Epoch {epoch+1}/{configs.HINT_STAGE1_EPOCHS}", unit="batch")
        for inputs, _ in pbar:
            inputs = inputs.to(configs.DEVICE)
            optimizer_s1.zero_grad()
            with torch.no_grad(): _, teacher_hint = teacher_model(inputs)
            _, student_hint = student_s1(inputs)
            loss = criterion_hint(student_hint, teacher_hint)
            loss.backward()
            optimizer_s1.step()
            pbar.set_postfix(mse_loss=loss.item())

    print("--- Stage 1 Finished. Preparing for Stage 2 (Distillation Training) ---")

    # --- STAGE 2: Train with Distillation Loss ---
    # The student model is the one we just pre-trained in Stage 1
    student_s2 = student_s1
    # Unfreeze all layers for Stage 2
    for param in student_s2.parameters():
        param.requires_grad = True
        
    criterion_kd = losses.DistillationLoss(configs.KD_ALPHA, temperature=configs.KD_TEMPERATURE)
    optimizer_s2 = optim.SGD(student_s2.parameters(), lr=configs.LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler_s2 = optim.lr_scheduler.StepLR(optimizer_s2, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    # The teacher for Stage 2 does not need to be a hint model, but it's fine to reuse it.
    
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Stage 2, Epoch {epoch+1}/{configs.EPOCHS} ---")
        student_s2.train()
        pbar = tqdm(train_loader, desc=f"Stage 2, Epoch {epoch+1}/{configs.EPOCHS}", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(configs.DEVICE), labels.to(configs.DEVICE)
            optimizer_s2.zero_grad()
            with torch.no_grad(): teacher_logits, _ = teacher_model(inputs)
            student_logits, _ = student_s2(inputs)
            loss = criterion_kd(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer_s2.step()
        scheduler_s2.step()
    
    # --- Save Final Model ---
    save_path = os.path.join(configs.CHECKPOINT_DIR, f"{experiment_name}_best.pth.tar")
    torch.save(student_s2.state_dict(), save_path)
    print(f"\n--- Training Finished ---")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()