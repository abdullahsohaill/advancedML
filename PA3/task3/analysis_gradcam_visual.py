# analysis_gradcam.py (Corrected for tuple output)

import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.grad_cam import GradCAM

# Import our modular components
import configs
from models_with_hints import VGGWithHint
from models_with_embeddings import VGGWithEmbedding

# --- THIS IS THE FIX: A simple wrapper class ---
class GradCamModelWrapper(nn.Module):
    """
    An adapter to make our tuple-returning models compatible with the grad-cam library,
    which expects a model to return a single tensor.
    """
    def __init__(self, model):
        super(GradCamModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Call the underlying model and get the tuple (logits, extra_data)
        output = self.model(x)
        # Return only the logits
        return output[0]

def main():
    """Main function to perform Grad-CAM analysis on all models."""
    print("--- Starting Grad-CAM Analysis (All Models) ---")
    
    # --- 1. Load All Models ---
    print("Loading Teacher and all 6 Student models...")
    
    models = {}
    
    # Teacher
    teacher_base = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    models['Teacher (VGG-16)'] = VGGWithEmbedding(base_model=teacher_base).to(configs.DEVICE)
    
    # Student model configurations
    student_configs = {
        "Baseline": ("student_hub_baseline_best.pth.tar", "base"),
        "LSR": ("student_hub_label_smoothing_best.pth.tar", "base"),
        "Logit Match": ("student_kd_lm_hub_fair_best.pth.tar", "base"),
        "DKD": ("student_kd_dkd_hub_best.pth.tar", "base"),
        "FitNets": ("student_fitnets_final_stage2_best.pth.tar", "hint"),
        "CRD": ("crd_new_best.pth.tar", "embedding")
    }

    for name, (path, model_type) in student_configs.items():
        base_vgg11 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
        
        if model_type == "base" or model_type == "embedding":
            model = VGGWithEmbedding(base_model=base_vgg11).to(configs.DEVICE)
        elif model_type == "hint":
            model = VGGWithHint(base_model=base_vgg11, hint_layer_index=8).to(configs.DEVICE)
            
        checkpoint_path = os.path.join(configs.CHECKPOINT_DIR, path)
        try:
            state = torch.load(checkpoint_path)
            if 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state)
            model.eval()
            models[name] = model
            print(f"Loaded {name} successfully.")
        except FileNotFoundError:
            print(f"ERROR: Could not find checkpoint for {name} at {checkpoint_path}. Skipping.")

    # --- 2. Select Target Layers for Grad-CAM ---
    target_layers = {}
    for name, model in models.items():
        target_layers[name] = model.features[-2]

    # --- 3. Load Images ---
    try:
        test_images_np = np.load('data/cifar-100-python/test_images.npy')
        test_labels_np = np.load('data/cifar-100-python/test_labels.npy')
    except FileNotFoundError:
        print("ERROR: test_images.npy or test_labels.npy not found.")
        print("Please run 'prepare_raw_data.py' first.")
        return
        
    test_dataset_raw = torch.utils.data.TensorDataset(torch.from_numpy(test_images_np), torch.from_numpy(test_labels_np))
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    image_indices = [100, 250, 450, 800]
    ssim_results = {name: [] for name in student_configs.keys()}

    # --- 4. Generate Visualizations ---
    num_images = len(image_indices)
    num_models = len(models)
    fig, axes = plt.subplots(num_images, num_models, figsize=(num_models * 3, num_images * 3.5))
    model_order = ['Teacher (VGG-16)'] + list(student_configs.keys())
    
    for i, img_idx in enumerate(image_indices):
        raw_img_np, label = test_dataset_raw[img_idx]
        raw_img_np = raw_img_np.numpy()
        input_tensor = test_transform(Image.fromarray(raw_img_np)).unsqueeze(0).to(configs.DEVICE)
        teacher_heatmap = None

        for j, name in enumerate(model_order):
            if name not in models: continue
            model = models[name]
            
            # --- APPLY THE WRAPPER ---
            model_for_cam = GradCamModelWrapper(model)
            
            cam = GradCAM(model=model_for_cam, target_layers=[target_layers[name]])
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
            
            visualization = show_cam_on_image(raw_img_np / 255.0, grayscale_cam, use_rgb=True)
            
            if name == 'Teacher (VGG-16)':
                teacher_heatmap = grayscale_cam
            else:
                score = ssim(teacher_heatmap, grayscale_cam, data_range=grayscale_cam.max() - grayscale_cam.min())
                ssim_results[name].append(score)

            ax = axes[i, j]
            ax.imshow(visualization)
            ax.axis('off')
            
            if i == 0:
                ax.set_title(name.replace(" ", "\n"), fontsize=10)
            
            # We still use the original model to get the prediction text
            logits, _ = model(input_tensor)
            pred_class = logits.argmax(dim=1).item()
            is_correct = "✓" if pred_class == label.item() else "✗"
            ax.text(1, 4, f"Pred: {pred_class} {is_correct}", color='white', backgroundcolor='black', fontsize=9)

    plt.tight_layout(pad=0.5, h_pad=1.0)
    fig_path = os.path.join("results", "grad_cam_comparison_all_models.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\nSaved Grad-CAM visualization grid to {fig_path}")
    
    # --- 5. Report SSIM Results ---
    print("\n--- Quantitative Similarity Results (SSIM) ---")
    print("Higher is more similar to the Teacher's heatmap.\n")
    
    avg_ssim_scores = {name: np.mean(scores) for name, scores in ssim_results.items()}
    sorted_scores = sorted(avg_ssim_scores.items(), key=lambda item: item[1], reverse=True)
    
    for name, avg_score in sorted_scores:
        print(f"Average SSIM for {name}: {avg_score:.4f}")

if __name__ == '__main__':
    main()