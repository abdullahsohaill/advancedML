# discover_hard_set.py

import torch
from tqdm import tqdm
import os
import numpy as np

# Import our modular components
import configs
from models_with_embeddings import VGGWithEmbedding
import data_loader

def main():
    """Finds and saves the indices of images that the baseline student gets wrong
    but the teacher gets right."""
    print("--- Discovering 'Hard Set' for Analysis ---")

    # --- 1. Load Models ---
    print("Loading Teacher and Baseline Student...")
    
    # Teacher
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    teacher_model.eval()

    # Independent Student (Baseline)
    baseline_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    independent_student = VGGWithEmbedding(base_model=baseline_base).to(configs.DEVICE)
    baseline_path = os.path.join(configs.CHECKPOINT_DIR, "student_hub_baseline_best.pth.tar")
    independent_student.load_state_dict(torch.load(baseline_path)['state_dict'])
    independent_student.eval()

    # --- 2. Load Test Data ---
    _, test_loader = data_loader.get_cifar100_loaders()
    
    # --- 3. Find Hard Images ---
    hard_image_indices = []
    current_index = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Finding Hard Images", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            teacher_logits, _ = teacher_model(inputs)
            teacher_preds = torch.argmax(teacher_logits, dim=1)
            
            student_logits, _ = independent_student(inputs)
            student_preds = torch.argmax(student_logits, dim=1)
            
            # Find where the teacher is correct AND the student is incorrect
            teacher_correct = (teacher_preds == labels)
            student_incorrect = (student_preds != labels)
            
            hard_mask = teacher_correct & student_incorrect
            
            # Get the global indices of these images
            batch_indices = torch.where(hard_mask)[0]
            for idx in batch_indices:
                hard_image_indices.append(current_index + idx.item())
            
            current_index += inputs.shape[0]

    print(f"\nFound {len(hard_image_indices)} 'hard' images.")
    
    # --- 4. Save the Indices ---
    save_path = os.path.join("results", "hard_set_indices.npy")
    np.save(save_path, np.array(hard_image_indices))
    print(f"Saved hard set indices to {save_path}")

if __name__ == '__main__':
    main()