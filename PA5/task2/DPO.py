import os
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only" 
DATASET_ID = "intel/orca_dpo_pairs"         
RUN_NAME = f"dpo_smollm2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"./models/{RUN_NAME}"
LOG_FILE = f"./logs/{RUN_NAME}.log"

# Hardware Settings (RTX 4090)
TORCH_DTYPE = torch.bfloat16
ATTN_IMPL = "sdpa" # Stable on Windows

# Hyperparameters
BATCH_SIZE = 8
GRAD_ACCUM = 8
LEARNING_RATE = 1e-6
BETA = 0.05
NUM_EPOCHS = 3

def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    )
    logging.info(f"Initialized Run: {RUN_NAME}")

def plot_training_history(log_history, save_path):
    """
    Generates a loss curve from the trainer's log history for the report.
    """
    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []
    rewards = []
    reward_steps = []

    for entry in log_history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
        # DPO logs rewards often (chosen/rejected rewards)
        if 'rewards/chosen' in entry:
            rewards.append(entry['rewards/chosen'])
            reward_steps.append(entry['step'])

    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue')
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red')
    plt.title('DPO Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Rewards (If available)
    if rewards:
        plt.subplot(1, 2, 2)
        plt.plot(reward_steps, rewards, label='Chosen Reward', color='green')
        plt.title('Reward Progression')
        plt.xlabel('Steps')
        plt.ylabel('Reward Score')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, "training_graphs.png")
    plt.savefig(plot_file)
    logging.info(f"Graphs saved to {plot_file}")
    plt.close()

def format_data(sample):
    system = sample.get('system', '')
    question = sample.get('question', '')
    if system:
        prompt = f"System: {system}\nUser: {question}\nAssistant:"
    else:
        prompt = f"User: {question}\nAssistant:"
    return {
        "prompt": prompt,
        "chosen": sample["chosen"],
        "rejected": sample["rejected"]
    }

def run_dpo():
    setup_logging()
    
    logging.info(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    
    # Format Dataset
    original_columns = dataset.column_names
    dataset = dataset.map(format_data, remove_columns=original_columns)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    logging.info(f"Loading SFT Baseline Model: {MODEL_ID}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=TORCH_DTYPE,
            attn_implementation=ATTN_IMPL,
            device_map="auto"
        )
    except Exception as e:
        logging.error(f"FATAL ERROR: Could not load model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO Configuration
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_length=1024,
        max_prompt_length=512,
        num_train_epochs=NUM_EPOCHS,
        
        # Updated TRL params
        eval_strategy="steps", 
        report_to="none", # DISABLES WANDB
        
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        bf16=True,
        run_name=RUN_NAME,
        remove_unused_columns=False
    )

    logging.info("Initializing DPOTrainer...")
    
    # UPDATED: Using 'processing_class' per new documentation
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer, 
    )

    logging.info("Starting DPO Training...")
    dpo_trainer.train()
    
    # Save Model & Metrics
    logging.info(f"Saving final model to {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Generate & Save Graphs locally
    logging.info("Generating training graphs...")
    plot_training_history(dpo_trainer.state.log_history, OUTPUT_DIR)
    
    logging.info("DPO Training Complete.")

if __name__ == "__main__":
    run_dpo()