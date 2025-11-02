# configs.py

import torch

# -- Environment --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# -- Dataset --
DATASET_PATH = "./data"
NUM_CLASSES = 100

# -- Training Hyperparameters --
BATCH_SIZE = 256  # High batch size for RTX 4090
EPOCHS = 50       # Number of epochs for baseline training
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULER_STEP_SIZE = 15
LR_SCHEDULER_GAMMA = 0.1

KD_TEMPERATURE = 4.0
KD_ALPHA = 0.3 # Weight for the hard loss (Cross-Entropy)
KD_LEARNING_RATE = 0.1

LABEL_SMOOTHING_VALUE = 0.1

DKD_BETA = 0.5   # Weight for the TCKD loss
DKD_GAMMA = 1.0  # Weight for the NCKD loss

HINT_STAGE1_EPOCHS = 20

CRD_LAMBDA = 1.0
CRD_CONTRASTIVE_TEMP = 0.1

# -- Dataloader --
NUM_WORKERS = 8 # Optimal for your CPU/GPU combo
PIN_MEMORY = True

# -- Model Saving --
CHECKPOINT_DIR = "./checkpoints"