import torch
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
REWARD_MODEL_PATH = "../models/rm_smollm2_run_20251130_192743"

DATASET_ID = "intel/orca_dpo_pairs"
RUN_NAME = f"grpo_smollm2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"./models/{RUN_NAME}"

# --- HYPERPARAMETERS ---
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-7,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    seed=42,
    num_generations=4, 
    max_completion_length=1024,
    max_prompt_length=512,
    beta=0.1,
    max_steps=100, # Stops after 60 steps
    logging_steps=5,
    save_steps=50,
    report_to="none",
)

def build_dataset():
    """Builds dataset, returning only the query text."""
    ds = load_dataset(DATASET_ID, split="train")
    
    # --- THE FIX IS HERE ---
    # GRPOTrainer strictly requires a column named 'prompt'
    ds = ds.rename_column("question", "prompt") 
    # -----------------------
    
    ds = ds.filter(lambda x: len(x["prompt"]) > 10) 
    return ds

def run_grpo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Models and Tokenizer
    print(f"Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto")
    
    # 2. Dataset
    dataset = build_dataset()

    # 3. Trainer
    print("Initializing GRPOTrainer...")
    grpo_trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_model,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # 4. Train
    print(f"Starting GRPO Training (CAPPED AT {config.max_steps} STEPS)...")
    grpo_trainer.train()

    # 5. Save
    print(f"Saving GRPO Model to {OUTPUT_DIR}...")
    grpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("GRPO Training Complete.")

if __name__ == "__main__":
    run_grpo()