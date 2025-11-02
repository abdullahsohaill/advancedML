# discover_easy_set.py

import torch
from tqdm import tqdm
import os
import numpy as np

# Import our modular components
import configs
from models_with_embeddings import VGGWithEmbedding
import data_loader

def main():
    """Finds and saves the indices of images that BOTH the baseline student AND
    the teacher get right."""
    print("--- Discovering 'Easy Set' for Analysis ---")

    # --- 1. Load Models ---
    print("Loading Teacher and Baseline Student...")
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    teacher_model.eval()

    baseline_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    independent_student = VGGWithEmbedding(base_model=baseline_base).to(configs.DEVICE)
    baseline_path = os.path.join(configs.CHECKPOINT_DIR, "student_hub_baseline_best.pth.tar")
    independent_student.load_state_dict(torch.load(baseline_path)['state_dict'])
    independent_student.eval()

    # --- 2. Load Test Data ---
    _, test_loader = data_loader.get_cifar100_loaders()
    
    # --- 3. Find Easy Images ---
    easy_image_indices = []
    current_index = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Finding Easy Images", unit="batch")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(configs.DEVICE), labels.to(configs.DEVICE)
            
            teacher_logits, _ = teacher_model(inputs)
            teacher_preds = torch.argmax(teacher_logits, dim=1)
            
            student_logits, _ = independent_student(inputs)
            student_preds = torch.argmax(student_logits, dim=1)
            
            # --- THE ONLY CHANGE IS HERE ---
            # Find where the teacher is correct AND the student is ALSO correct
            teacher_correct = (teacher_preds == labels)
            student_correct = (student_preds == labels) # Changed from incorrect
            
            easy_mask = teacher_correct & student_correct
            
            batch_indices = torch.where(easy_mask)[0]
            for idx in batch_indices:
                easy_image_indices.append(current_index + idx.item())
            
            current_index += inputs.shape[0]

    print(f"\nFound {len(easy_image_indices)} 'easy' images.")
    
    # --- 4. Save the Indices ---
    save_path = os.path.join("results", "easy_set_indices.npy")
    np.save(save_path, np.array(easy_image_indices))
    print(f"Saved easy set indices to {save_path}")

if __name__ == '__main__':
    main()