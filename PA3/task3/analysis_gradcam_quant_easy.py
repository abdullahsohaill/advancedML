# analysis_gradcam_cosine.py (Modified for Easy Set Analysis)

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam.grad_cam import GradCAM

# Import our modular components and the Grad-CAM wrapper
import configs
from models_with_hints import VGGWithHint
from models_with_embeddings import VGGWithEmbedding
from analysis_gradcam_visual import GradCamModelWrapper # Re-using the wrapper

def main():
    """
    Quantifies localization similarity for ALL distilled models against a baseline,
    running specifically on a pre-computed "easy set" of images where the
    baseline fails but the teacher succeeds.
    """
    print("--- Starting Grad-CAM Cosine Similarity Analysis (on Easy Set) ---")
    
    # --- 1. Load All Models ---
    print("Loading Teacher, Baseline, and all 5 Distilled Student models...")
    
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
    
    # Load all distilled student models
    distilled_students = {}
    student_configs = {
        "LSR": ("student_hub_label_smoothing_best.pth.tar", "base"),
        "Logit Match": ("student_kd_lm_hub_fair_best.pth.tar", "base"),
        "FitNets": ("student_fitnets_final_stage2_best.pth.tar", "hint"),
        "DKD": ("student_kd_dkd_hub_best.pth.tar", "base"),
        "CRD": ("crd_new_best.pth.tar", "embedding")
    }
    for name, (path, model_type) in student_configs.items():
        base_vgg11 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=False)
        if model_type == "base" or model_type == "embedding": model = VGGWithEmbedding(base_model=base_vgg11).to(configs.DEVICE)
        elif model_type == "hint": model = VGGWithHint(base_model=base_vgg11, hint_layer_index=8).to(configs.DEVICE)
        checkpoint_path = os.path.join(configs.CHECKPOINT_DIR, path)
        state = torch.load(checkpoint_path)
        if 'state_dict' in state: model.load_state_dict(state['state_dict'])
        else: model.load_state_dict(state)
        model.eval()
        distilled_students[name] = model

    # --- 2. Prepare Data Loader (Using the Easy Set) ---
    print("\nLoading the 'Easy Set' for analysis...")
    try:
        easy_set_indices = np.load(os.path.join("results", "easy_set_indices.npy"))
        print(f"Loaded {len(easy_set_indices)} indices for the easy set.")
    except FileNotFoundError:
        print("ERROR: easy_set_indices.npy not found. Please run 'discover_easy_set.py' first.")
        return

    test_images_np = np.load('data/cifar-100-python/test_images.npy')
    
    # Create a subset of the data using these indices
    easy_set_images_np = test_images_np[easy_set_indices]

    # The dataset and loader now only contain the easy images
    easy_dataset = torch.utils.data.TensorDataset(torch.from_numpy(easy_set_images_np))
    test_loader = torch.utils.data.DataLoader(easy_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    # --- 3. Setup Grad-CAM for all models ---
    teacher_cam_model = GradCamModelWrapper(teacher_model)
    independent_cam_model = GradCamModelWrapper(independent_student)
    
    cam_teacher = GradCAM(model=teacher_cam_model, target_layers=[teacher_model.features[-2]])
    cam_independent = GradCAM(model=independent_cam_model, target_layers=[independent_student.features[-2]])

    cams_distilled = {}
    for name, model in distilled_students.items():
        distilled_cam_model = GradCamModelWrapper(model)
        cams_distilled[name] = GradCAM(model=distilled_cam_model, target_layers=[model.features[-2]])

    # --- 4. Run the Analysis ---
    win_counters = {name: 0 for name in distilled_students.keys()}
    total_images = 0

    pbar = tqdm(test_loader, desc="Calculating CAM Similarities on Easy Set", unit="batch")
    for (images_np,) in pbar:
        for i in range(images_np.shape[0]):
            raw_img_np = images_np[i].numpy()
            input_tensor = test_transform(Image.fromarray(raw_img_np)).unsqueeze(0).to(configs.DEVICE)

            cam_t = cam_teacher(input_tensor=input_tensor)[0, :]
            cam_i = cam_independent(input_tensor=input_tensor)[0, :]

            vec_t = torch.from_numpy(cam_t).flatten()
            vec_i = torch.from_numpy(cam_i).flatten()
            sim_ti = F.cosine_similarity(vec_t, vec_i, dim=0)

            for name, cam_distilled in cams_distilled.items():
                cam_d = cam_distilled(input_tensor=input_tensor)[0, :]
                vec_d = torch.from_numpy(cam_d).flatten()
                sim_td = F.cosine_similarity(vec_t, vec_d, dim=0)

                if sim_td > sim_ti:
                    win_counters[name] += 1
            
            total_images += 1
    
    # --- 5. Report the Final Results ---
    print("\n--- Localization Knowledge Transfer Analysis (on Easy Set) ---")
    print(f"Analyzed {total_images} 'easy' images.")
    print("Percentage of these images where the distilled model's CAM was more similar to the Teacher's than the Baseline's was:")
    
    win_percentages = {name: (count / total_images) * 100 for name, count in win_counters.items()}
    sorted_percentages = sorted(win_percentages.items(), key=lambda item: item[1], reverse=True)
    
    print("\nModel                | Win Percentage")
    print("---------------------|---------------")
    for name, percentage in sorted_percentages:
        print(f"{name:<20} | {percentage:.2f}%")
    print("\n(A result > 50% indicates successful localization knowledge transfer on challenging images.)")

if __name__ == '__main__':
    main()