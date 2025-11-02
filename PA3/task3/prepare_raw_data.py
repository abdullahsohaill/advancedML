# prepare_raw_data.py

import torchvision.datasets as datasets
import numpy as np
import os

def create_raw_cifar_files():
    """
    Loads the CIFAR-100 test set and saves its raw image and label data
    as numpy arrays for use in visualization scripts.
    """
    print("Loading raw CIFAR-100 test set from torchvision...")
    
    # Load the dataset object without any transforms to access the raw data
    test_set = datasets.CIFAR100(root='./data', train=False, download=True)

    # The raw data is stored in the .data and .targets attributes as numpy arrays/lists
    test_images = test_set.data  # This is a numpy array of shape (10000, 32, 32, 3)
    test_labels = np.array(test_set.targets) # This is a list, so we convert it

    # Define the path where the analysis script expects the files
    save_path = './data/cifar-100-python/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    # Save the numpy arrays to disk
    np.save(os.path.join(save_path, 'test_images.npy'), test_images)
    print(f"Saved test_images.npy with shape: {test_images.shape}")

    np.save(os.path.join(save_path, 'test_labels.npy'), test_labels)
    print(f"Saved test_labels.npy with shape: {test_labels.shape}")

    print("\nRaw data files created successfully!")
    print("You can now re-run the 'analysis_gradcam.py' script.")

if __name__ == '__main__':
    create_raw_cifar_files()