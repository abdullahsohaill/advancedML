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
RUN_NAME = f"ppo_dense_smollm2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = f"./models/{RUN_NAME}"

# --- HYPERPARAMETERS ---
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=1,
    seed=42,
    log_with=None,
)
MAX_STEPS = 100

# --- CUSTOM TRAINER FOR DENSE REWARDS (FIXED) ---
class DensePPOTrainer(PPOTrainer):
    """
    Extended PPOTrainer that supports token-level (dense) rewards.
    """
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        # 1. Calculate the standard KL penalty
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone()

        # 2. Inject our Custom Dense Rewards
        # We ignore 'scores' (dummy zeros) and use self.current_dense_rewards
        
        for i, dense_tensor in enumerate(self.current_dense_rewards):
            response_length = len(dense_tensor)
            
            # Add the dense reward to the end of the sequence
            if response_length > 0:
                rewards[i, -response_length:] += dense_tensor.to(rewards.device)
            
        # THE FIX: Return 3 values: total rewards, non-score rewards, and raw KL
        return rewards, non_score_reward, kl 

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

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def run_ppo_dense():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Models
    print(f"Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto")
    
    model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. Dataset & Trainer
    dataset = build_dataset(tokenizer)
    print("Initializing DensePPOTrainer...")
    
    ppo_trainer = DensePPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # 3. Manual Training Loop
    print(f"Starting DENSE PPO Loop (CAPPED AT {MAX_STEPS} STEPS)...")
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
            rewards_bfloat16 = reward_model(**rm_inputs).logits.squeeze(-1)
            rewards_float32 = rewards_bfloat16.to(torch.float32)
            
            # --- DENSE REWARD CALCULATION ---
            dense_rewards_list = []
            for i in range(len(rewards_float32)):
                scalar_reward = rewards_float32[i]
                response_len = response_tensors[i].shape[0]
                
                # Distribute score: score / length
                token_reward = scalar_reward / max(1, response_len)
                dense_tensor = torch.full((response_len,), token_reward, dtype=torch.float32)
                dense_rewards_list.append(dense_tensor)
            
            # Inject into trainer
            ppo_trainer.current_dense_rewards = dense_rewards_list
            # --------------------------------

        # C. Update
        # Pass dummy TENSORS to satisfy safety checks
        dummy_scores = [torch.tensor(0.0, device=model.current_device) for _ in range(len(dense_rewards_list))]
        
        stats = ppo_trainer.step(query_tensors, response_tensors, dummy_scores)
        
        if step % 5 == 0:
            print(f"\nStep {step}: Mean Reward (Sum): {torch.tensor(rewards_float32).mean().item():.4f}")
            # Log sum of dense rewards (approx equal to original scalar)
            ppo_trainer.log_stats(stats, batch, [r.sum() for r in dense_rewards_list])

    # 4. Save
    print(f"Saving Dense PPO Model to {OUTPUT_DIR}...")
    ppo_trainer.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Dense PPO Training Complete.")

if __name__ == "__main__":
    run_ppo_dense()