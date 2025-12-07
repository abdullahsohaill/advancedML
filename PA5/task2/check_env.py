import trl
import transformers
import inspect

# These imports should now point to the STABLE versions
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

print("="*40)
print(f"TRL Version: {trl.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print("="*40)

print("\n[STABLE PPOConfig Arguments]")
# Prints the exact arguments your version of PPOConfig accepts
print(inspect.signature(PPOConfig.__init__))

print("\n[STABLE PPOTrainer Arguments]")
# Prints the exact arguments your version of PPOTrainer accepts
print(inspect.signature(PPOTrainer.__init__))

print("\n[Model Wrapper Inspection]")
# This should now succeed without error
try:
    # Note: Loading the model can be slow, but it's the only way to check attributes
    print("Loading model to check compatibility...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained("HuggingFaceTB/smollm2-135M-SFT-Only")
    print(f"Model loaded successfully.")
    print(f"Has generation_config? {hasattr(model, 'generation_config')}")
except Exception as e:
    print(f"Model load or compatibility error: {e}")

print("="*40)