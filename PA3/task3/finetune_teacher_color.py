# finetune_teacher_color.py

import torch
import torch.nn as nn
import torch.optim as optim
import os

import configs
import data_loader
import trainers
import utils
from models_with_embeddings import VGGWithEmbedding

def main():
    """Step 1: Fine-tune the pre-trained teacher to be color-invariant."""
    
    # --- 1. Load Teacher and Data ---
    print("Loading pre-trained teacher and aggressive color jitter data...")
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    
    # Use the new loader with aggressive training jitter
    train_loader_jitter = data_loader.get_cifar100_color_jitter_loaders(for_training=True)
    test_loader_jitter = data_loader.get_cifar100_color_jitter_loaders(for_training=False)

    # --- 2. Setup for Fine-tuning ---
    # We are just fine-tuning, so a simple loss and a low learning rate is best.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    
    # Fine-tune for a small number of epochs
    finetune_epochs = 15
    
    print(f"--- Starting to Fine-tune Teacher for {finetune_epochs} epochs ---")
    
    # --- 3. Fine-tuning Loop ---
    for epoch in range(finetune_epochs):
        print(f"\n--- Fine-tuning Epoch {epoch+1}/{finetune_epochs} ---")
        
        # Use our standard trainer, as this is a simple supervised task
        train_loss, train_acc = trainers.train_epoch(
            teacher_model, train_loader_jitter, criterion, optimizer, configs.DEVICE
        )
        # Evaluate on the color-jittered test set to monitor progress
        test_loss, test_acc = trainers.evaluate(
            teacher_model, test_loader_jitter, criterion, configs.DEVICE
        )
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Acc (Jitter): {train_acc:.2f}% | Test Acc (Jitter): {test_acc:.2f}%")

    # --- 4. Save the Final Model ---
    save_path = os.path.join(configs.CHECKPOINT_DIR, "teacher_color_invariant.pth.tar")
    # Save the whole wrapped model for consistency
    torch.save(teacher_model.state_dict(), save_path)
    
    print("\n--- Fine-tuning Finished ---")
    print(f"Color-invariant teacher saved to {save_path}")

if __name__ == '__main__':
    main()