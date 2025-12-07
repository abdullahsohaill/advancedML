import os
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# --- CONFIGURATION ---
# We start from the SFT model again. The RM is a separate fork of the base model.
MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only" 
DATASET_ID = "intel/orca_dpo_pairs"         
RUN_NAME = f"rm_smollm2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"./models/{RUN_NAME}"
LOG_FILE = f"./logs/{RUN_NAME}.log"

# Hardware Settings
TORCH_DTYPE = torch.bfloat16
ATTN_IMPL = "sdpa"

# Hyperparameters for Reward Modeling
BATCH_SIZE = 32  #RMs are classifiers, so they use less memory than generation
GRAD_ACCUM = 2
LEARNING_RATE = 2e-5 # RMs can handle higher LR than DPO
NUM_EPOCHS = 1 
MAX_LENGTH = 1024

def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    )
    logging.info(f"Initialized Reward Model Run: {RUN_NAME}")

def plot_rm_history(log_history, save_path):
    """Plots accuracy and loss for the Reward Model."""
    steps = []
    losses = []
    accuracies = []
    
    for entry in log_history:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
        if 'eval_accuracy' in entry:
            # TRL RewardTrainer computes accuracy (how often preferred > rejected)
            accuracies.append(entry['eval_accuracy'])

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, label='Train Loss', color='purple')
    plt.title('Reward Model Loss')
    plt.xlabel('Steps')
    plt.legend()
    plt.grid(True)

    if accuracies:
        # We might have fewer eval points than train points, so just plot what we have
        eval_steps = [x * (steps[-1] // len(accuracies)) for x in range(1, len(accuracies) + 1)]
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, accuracies, label='Accuracy', color='orange')
        plt.title('Preference Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy (Preferred > Rejected)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rm_training_graphs.png"))
    plt.close()

def format_data(sample):
    """
    Reward Trainer expects columns: 'input_ids_chosen', 'attention_mask_chosen', ...
    The trainer handles the tokenization if we give it 'chosen' and 'rejected' text lists.
    We just need to format the strings properly first.
    """
    system = sample.get('system', '')
    question = sample.get('question', '')
    
    # Format: User query -> Assistant response
    if system:
        base_prompt = f"System: {system}\nUser: {question}\nAssistant: "
    else:
        base_prompt = f"User: {question}\nAssistant: "
        
    return {
        "chosen": base_prompt + sample["chosen"],
        "rejected": base_prompt + sample["rejected"]
    }

def run_reward_modeling():
    setup_logging()
    
    logging.info(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    
    # 1. Format text
    # The RewardTrainer class is smart. It needs 'chosen' and 'rejected' text columns.
    dataset = dataset.map(format_data, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    logging.info(f"Loading Model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL: We load AutoModelForSequenceClassification, NOT CausalLM
    # num_labels=1 means it outputs a single scalar score
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1, 
        dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPL,
        device_map="auto"
    )
    
    # Assign pad token to model config to avoid errors
    model.config.pad_token_id = tokenizer.pad_token_id

    # 2. Configure Training
    training_args = RewardConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        max_length=MAX_LENGTH,
        
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        bf16=True,
        report_to="none",
        run_name=RUN_NAME,
        remove_unused_columns=False,
        # Important for RMs: center rewards so mean is 0
        center_rewards_coefficient=0.01 
    )

    logging.info("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    logging.info("Starting Reward Model Training...")
    trainer.train()
    
    logging.info(f"Saving Reward Model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    plot_rm_history(trainer.state.log_history, OUTPUT_DIR)
    logging.info("Reward Model Training Complete.")

if __name__ == "__main__":
    run_reward_modeling()