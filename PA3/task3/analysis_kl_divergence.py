# analysis_kl_divergence.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import our modular components
import configs
import data_loader
from models_with_hints import VGGWithHint
from models_with_embeddings import VGGWithEmbedding

def calculate_kl_divergence(teacher_logits, student_logits, temperature):
    """
    Calculates the KL Divergence between teacher and student output distributions.
    """
    # Soften the distributions
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # KL Divergence expects log-probabilities for the input and probabilities for the target.
    # The 'batchmean' reduction averages the loss over the batch.
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # The scaling factor used in the original paper to balance gradients.
    # We include it here for consistency with the training loss formulation.
    return kl_div * (temperature ** 2)

def main():
    """Main function to perform KL divergence analysis."""
    print("--- Starting KL Divergence Analysis ---")
    
    # --- 1. Load Data ---
    _, test_loader = data_loader.get_cifar100_loaders()

    # --- 2. Load Models ---
    print("Loading all trained models...")
    
    # Teacher Model
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    teacher_model = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE) # Using wrapper for consistent output
    teacher_model.eval()

    # Dictionary to hold all student models
    student_models = {}
    
    # Define model paths and architectures
    model_configs = {
        "Baseline": ("student_hub_baseline_best.pth.tar", "base"),
        "LSR": ("student_hub_label_smoothing_best.pth.tar", "base"),
        "Logit Match": ("student_kd_lm_hub_fair_best.pth.tar", "base"),
        "DKD": ("student_kd_dkd_hub_best.pth.tar", "base"),
        "FitNets": ("student_fitnets_final_stage2_best.pth.tar", "hint"),
        "CRD": ("crd_new_best.pth.tar", "embedding")
    }

    for name, (path, model_type) in model_configs.items():
        # Instantiate the correct base architecture (untrained)
        base_vgg11 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
        
        # Wrap the model if necessary
        if model_type == "base":
            model = VGGWithEmbedding(base_model=base_vgg11).to(configs.DEVICE) # Wrap for consistent output
        elif model_type == "hint":
            model = VGGWithHint(base_model=base_vgg11, hint_layer_index=8).to(configs.DEVICE)
        elif model_type == "embedding":
            model = VGGWithEmbedding(base_model=base_vgg11).to(configs.DEVICE)
            
        # Load the saved weights
        checkpoint_path = os.path.join(configs.CHECKPOINT_DIR, path)
        # For FitNets and CRD, the state_dict is saved directly. Others are in a checkpoint dict.
        try:
            state = torch.load(checkpoint_path)
            if 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state)
        except FileNotFoundError:
            print(f"ERROR: Could not find checkpoint for {name} at {checkpoint_path}. Skipping.")
            continue
            
        model.eval()
        student_models[name] = model
        print(f"Loaded {name} successfully.")
        
    # --- 3. Run Analysis ---
    print("\nCalculating average KL Divergence over the test set...")
    
    kl_div_totals = {name: 0.0 for name in student_models.keys()}
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Analysis", unit="batch")
        for inputs, _ in pbar:
            inputs = inputs.to(configs.DEVICE)
            
            teacher_logits, _ = teacher_model(inputs)
            
            for name, model in student_models.items():
                student_logits, _ = model(inputs)
                kl_div = calculate_kl_divergence(teacher_logits, student_logits, configs.KD_TEMPERATURE)
                kl_div_totals[name] += kl_div.item()
            
            num_batches += 1

    # Calculate averages
    avg_kl_divs = {name: total / num_batches for name, total in kl_div_totals.items()}

    # --- 4. Report Results ---
    df = pd.DataFrame(list(avg_kl_divs.items()), columns=['Model', 'Average KL Divergence'])
    df = df.sort_values(by='Average KL Divergence', ascending=True).reset_index(drop=True)
    
    print("\n--- KL Divergence Analysis Results ---")
    print(df)

    # --- 5. Visualize Results ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    barplot = sns.barplot(x='Average KL Divergence', y='Model', data=df, palette='viridis')
    
    plt.title('Average KL Divergence (Student vs. Teacher)', fontsize=16)
    plt.xlabel('KL Divergence (Lower is Better)', fontsize=12)
    plt.ylabel('Student Model', fontsize=12)
    
    # Add values on the bars
    for index, value in enumerate(df['Average KL Divergence']):
        plt.text(value, index, f' {value:.4f}', va='center')
        
    # Save the figure for the report
    fig_path = os.path.join("results", "kl_divergence_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {fig_path}")

if __name__ == '__main__':
    main()