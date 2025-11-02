# evaluate_teacher.py (Corrected with torch.hub)

import torch
import torch.nn as nn

# Import our modular components
import configs
import data_loader
import trainers

def main():
    """Main function to evaluate the teacher model using torch.hub."""
    
    print("--- Evaluating Teacher Model (VGG-16) ---")

    # --- 1. Load Data ---
    print("Loading CIFAR-100 test dataset...")
    # We only need the test_loader for evaluation
    _, test_loader = data_loader.get_cifar100_loaders()
    print("Dataset loaded successfully.")

    # --- 2. Load Pretrained Teacher Model via Torch Hub ---
    print("Loading pretrained VGG-16 teacher from torch.hub...")
    
    # This single line handles downloading the model architecture and the fine-tuned weights
    teacher_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", # Repository owner and name
        "cifar100_vgg16_bn",             # The specific model for CIFAR-100
        pretrained=True                  # Load the pretrained weights
    ).to(configs.DEVICE)

    print("Teacher model loaded successfully.")
    
    # Define the loss function (needed for the evaluate function)
    criterion = nn.CrossEntropyLoss()

    # --- 3. Evaluate the Model ---
    print(f"Evaluating on {configs.DEVICE}...")
    # We can reuse our generic evaluate function perfectly
    test_loss, test_acc = trainers.evaluate(
        teacher_model, test_loader, criterion, configs.DEVICE
    )

    print("\n--- Evaluation Finished ---")
    print(f"Teacher Model Test Loss: {test_loss:.4f}")
    print(f"Teacher Model Test Accuracy: {test_acc:.2f}%")
    
    # Save the teacher model for later use in distillation
    torch.save(teacher_model.state_dict(), f"{configs.CHECKPOINT_DIR}/teacher_vgg16_cifar100.pth")
    print(f"Teacher model saved to {configs.CHECKPOINT_DIR}/teacher_vgg16_cifar100.pth")


if __name__ == '__main__':
    main()