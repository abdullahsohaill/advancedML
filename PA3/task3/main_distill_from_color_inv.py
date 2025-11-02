# main_distill_from_color_inv.py

import torch
import torch.optim as optim
import os
import argparse # Use argparse to select the KD method

import configs
import data_loader
import trainers_kd
import utils
import losses
from models_with_embeddings import VGGWithEmbedding

def main(method):
    """
    Step 2 (Expanded): Distill from the color-invariant teacher using a specified method (lm, dkd, or crd).
    """
    
    experiment_name = f"student_{method}_from_color_inv_teacher"
    print(f"\n--- Starting Distillation using method: {method.upper()} ---")

    # --- 1. Load Data (REGULAR augmentations) ---
    train_loader, _ = data_loader.get_cifar100_loaders()

    # --- 2. Load Models ---
    student_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    student_model = VGGWithEmbedding(base_model=student_base).to(configs.DEVICE)

    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=False)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    teacher_path = os.path.join(configs.CHECKPOINT_DIR, "teacher_color_invariant.pth.tar")
    teacher_model.load_state_dict(torch.load(teacher_path))
    teacher_model.eval()

    # --- 3. Select Loss Function and Trainer based on method ---
    if method == 'lm':
        criterion_kd = losses.DistillationLoss(alpha=0.5, temperature=configs.KD_TEMPERATURE)
        train_function = trainers_kd.train_epoch_kd
    elif method == 'dkd':
        criterion_kd = losses.DKDLoss(beta=configs.DKD_BETA, gamma=configs.DKD_GAMMA, temperature=configs.KD_TEMPERATURE)
        train_function = trainers_kd.train_epoch_kd
    elif method == 'crd':
        criterion_kd = losses.CRDLoss(
            alpha=0.5, 
            kd_temp=configs.KD_TEMPERATURE,
            crd_temp=configs.CRD_CONTRASTIVE_TEMP,
            lambda_crd=configs.CRD_LAMBDA
        )
        train_function = trainers_kd.train_epoch_crd
    else:
        raise ValueError("Invalid method specified. Choose from 'lm', 'dkd', 'crd'.")

    optimizer = optim.SGD(student_model.parameters(), lr=configs.LEARNING_RATE, momentum=configs.MOMENTUM, weight_decay=configs.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configs.LR_SCHEDULER_STEP_SIZE, gamma=configs.LR_SCHEDULER_GAMMA)
    
    print(f"--- Training: {experiment_name} ---")

    # --- 4. Training Loop ---
    for epoch in range(configs.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{configs.EPOCHS} ---")
        train_function(student_model, teacher_model, train_loader, criterion_kd, optimizer, configs.DEVICE)
        scheduler.step()

    # --- 5. Save Final Model ---
    save_path = os.path.join(configs.CHECKPOINT_DIR, f"{experiment_name}_best.pth.tar")
    torch.save(student_model.state_dict(), save_path)
    print(f"\n--- Training Finished ---")
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run distillation from color-invariant teacher.')
    parser.add_argument('--method', type=str, required=True, choices=['lm', 'dkd', 'crd'],
                        help='Distillation method to use.')
    args = parser.parse_args()
    main(args.method)