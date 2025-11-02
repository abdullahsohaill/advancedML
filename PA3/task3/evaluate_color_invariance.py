# evaluate_color_invariance.py (Updated to include FitNets)

import torch
import torch.nn as nn
import os

import configs
import data_loader
import trainers
from models_with_embeddings import VGGWithEmbedding
from models_with_hints import VGGWithHint # Needed for FitNets

def main():
    """Step 3 (Expanded): Evaluate all relevant models, including FitNets, on the color-jittered test set."""
    
    print("--- Evaluating Color Invariance Performance (Full Comparison ft. FitNets) ---")
    
    # --- 1. Load the color-jittered test set ---
    test_loader_jitter = data_loader.get_cifar100_color_jitter_loaders(for_training=False)

    # --- 2. Load all models for comparison ---
    models_to_evaluate = {}
    
    # --- Group 1: Baselines (trained on REGULAR data) ---
    print("\nLoading baseline models (trained on regular data)...")
    student_configs_orig = {
        "Baseline": "student_hub_baseline_best.pth.tar",
        "Logit Match": "student_kd_lm_hub_fair_best.pth.tar",
        "DKD": "student_kd_dkd_hub_best.pth.tar",
        "CRD": "crd_new_best.pth.tar",
    }
    for name, path in student_configs_orig.items():
        base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
        model = VGGWithEmbedding(base_model=base).to(configs.DEVICE)
        model_path = os.path.join(configs.CHECKPOINT_DIR, path)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        models_to_evaluate[f'{name} (Original)'] = model

    # FitNets Original
    fitnets_orig_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    fitnets_orig_student = VGGWithHint(base_model=fitnets_orig_base, hint_layer_index=8).to(configs.DEVICE)
    fitnets_orig_path = os.path.join(configs.CHECKPOINT_DIR, "student_fitnets_final_stage2_best.pth.tar")
    fitnets_orig_student.load_state_dict(torch.load(fitnets_orig_path)['state_dict'])
    models_to_evaluate['FitNets (Original)'] = fitnets_orig_student
    
    # --- Group 2: Color-Distilled Models ---
    print("\nLoading color-distilled models...")
    student_configs_new = {
        "Logit Match": "student_lm_from_color_inv_teacher_best.pth.tar",
        "DKD": "student_dkd_from_color_inv_teacher_best.pth.tar",
        "CRD": "student_crd_from_color_inv_teacher_best.pth.tar",
    }
    for name, path in student_configs_new.items():
        base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
        model = VGGWithEmbedding(base_model=base).to(configs.DEVICE)
        model_path = os.path.join(configs.CHECKPOINT_DIR, path)
        model.load_state_dict(torch.load(model_path))
        models_to_evaluate[f'{name} (Color-Distilled)'] = model
        
    # FitNets New
    fitnets_new_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
    fitnets_new_student = VGGWithHint(base_model=fitnets_new_base, hint_layer_index=8).to(configs.DEVICE)
    fitnets_new_path = os.path.join(configs.CHECKPOINT_DIR, "student_fitnets_from_color_inv_teacher_best.pth.tar")
    fitnets_new_student.load_state_dict(torch.load(fitnets_new_path))
    models_to_evaluate['FitNets (Color-Distilled)'] = fitnets_new_student
    
    # --- Group 3: The Teacher (Performance Ceiling) ---
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=False)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    teacher_path = os.path.join(configs.CHECKPOINT_DIR, "teacher_color_invariant.pth.tar")
    teacher_model.load_state_dict(torch.load(teacher_path))
    models_to_evaluate['Color-Inv Teacher'] = teacher_model
    
    # --- 3. Run Evaluation ---
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, model in models_to_evaluate.items():
        print(f"\nEvaluating {name}...")
        model.eval()
        _, test_acc = trainers.evaluate(model, test_loader_jitter, criterion, configs.DEVICE)
        results[name] = test_acc

    # --- 4. Report Final Comparison ---
    print("\n\n--- Final Color Invariance Results (ft. FitNets) ---")
    print("Accuracy on Color-Jittered Test Set:\n")
    print(f"{'Model':<35} | {'Accuracy':<10} | {'Improvement':<10}")
    print("-" * 60)
    
    baselines_order = ['Baseline (Original)', 'Logit Match (Original)', 'DKD (Original)', 'CRD (Original)', 'FitNets (Original)']
    for name in baselines_order:
        print(f"{name:<35} | {results[name]:.2f}%      |")
    print("-" * 60)

    distilled_order = ['Logit Match', 'DKD', 'CRD', 'FitNets']
    for name in distilled_order:
        orig_name = f'{name} (Original)'
        new_name = f'{name} (Color-Distilled)'
        improvement = results[new_name] - results[orig_name]
        print(f"{new_name:<35} | {results[new_name]:.2f}%      | +{improvement:.2f}%")
    print("-" * 60)

    print(f"{'Color-Inv Teacher':<35} | {results['Color-Inv Teacher']:.2f}%      | (Ceiling)")

if __name__ == '__main__':
    main()