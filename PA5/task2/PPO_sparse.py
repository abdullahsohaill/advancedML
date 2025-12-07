import torch
import os
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead 
from datasets import load_dataset

# --- CONFIGURATION ---
MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
REWARD_MODEL_PATH = "./models/rm_smollm2_run_20251130_192743" 

DATASET_ID = "intel/orca_dpo_pairs"
RUN_NAME = f"ppo_smollm2_final_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"./models/{RUN_NAME}"

# --- HYPERPARAMETERS ---
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,                  # Use more of your 4090's VRAM
    mini_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size of 64
    ppo_epochs=1,                   # THE FIX: Prevents over-correction and is much faster
    seed=42,
    log_with=None,
)
MAX_STEPS = 100

# --- DATASET PREPARATION ---
def build_dataset(tokenizer):
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.rename_column("question", "query")
    ds = ds.filter(lambda x: len(x["query"]) > 10) 
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# --- SIMPLE COLLATOR (THE FIX) ---
def collator(data):
    # This just creates a dictionary of lists, which is what the trainer expects.
    # It does NOT create a padded tensor batch.
    return {key: [d[key] for d in data] for key in data[0]}

def run_ppo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Models and Tokenizer
    print(f"Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto")
    
    model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. Dataset & Trainer
    dataset = build_dataset(tokenizer)
    print("Initializing Stable PPOTrainer...")
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # 3. Manual Training Loop
    print(f"Starting PPO Loop (CAPPED AT {MAX_STEPS} STEPS)...")
    generation_kwargs = { "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 50 }

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=MAX_STEPS):
        if step >= MAX_STEPS:
            break
            
        query_tensors = batch["input_ids"]
        
        # A. Generate
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # B. Score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rm_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
        with torch.no_grad():
            # This returns a bfloat16 tensor
            rewards_bfloat16 = reward_model(**rm_inputs).logits.squeeze(-1)
            # --- THE GUARANTEED FIX IS HERE ---
            # Convert the entire tensor to float32 before creating the list
            rewards_float32 = rewards_bfloat16.to(torch.float32)
            # Create a list of float32 tensors for the trainer functions
            rewards_list = [score for score in rewards_float32]
            # ------------------------------------

        # C. Update using the corrected float32 list
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
        
        if step % 5 == 0:
            print(f"\nStep {step}: Mean Reward: {torch.tensor(rewards_list).mean().item():.4f}")
            # Pass the corrected float32 list to the buggy log function
            ppo_trainer.log_stats(stats, batch, rewards_list)

    # 4. Save
    print(f"Saving PPO Model to {OUTPUT_DIR}...")
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("PPO Training Complete.")

if __name__ == "__main__":
    run_ppo()