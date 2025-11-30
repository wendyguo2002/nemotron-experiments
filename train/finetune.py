import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from typing import Dict, Any

# --- Configuration ---
MODEL_ID = "nvidia/Nemotron-Mini-4B-Instruct" # Base Nemotron Model from Hugging Face
DATASET_PATH = "financial_synthetic_data.jsonl" # MUST be in the same directory
OUTPUT_DIR = "nemotron-mini-4b-stock-lora"
LORA_ADAPTER_NAME = "stock_sentiment_adapter"

# --- 1. Device Setup (Crucial for Mac M1) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Metal GPU (MPS) for training.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Falling back to CPU (will be very slow).")
    
# --- 2. Model and Tokenizer Loading ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for Causal LM
    print("✅ Tokenizer loaded.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, # Use float32 for stable MPS training
        device_map={'': device},  # Map model to the MPS device
        trust_remote_code=True
    )
    print(f"✅ Model '{MODEL_ID}' loaded on {device}.")

except Exception as e:
    print(f"❌ Error loading model or tokenizer: {e}")
    exit(1)


# --- 3. Data Loading and Formatting (FIXED to prevent NameError) ---

def format_example(example: Dict[str, Any], tok: AutoTokenizer) -> Dict[str, str]:
    """
    Formats a single instruction/response example into the Nemotron chat template.
    The tokenizer (tok) is explicitly passed to avoid multiprocessing errors.
    """
    headline = example['instruction']
    json_response = example['response']
    
    # Nemotron's recommended Instruction Template:
    # <extra_id_0>System {system prompt} <extra_id_1>User {prompt} <extra_id_1>Assistant\n
    system_prompt = "You are an expert financial analyst. Your task is to analyze the user's provided stock news headline and output a structured JSON object containing the company ticker, sentiment, event type, and expected stock impact."
    
    text = (
        f"<extra_id_0>System {system_prompt} <extra_id_1>User {headline} <extra_id_1>Assistant\n{json_response}{tok.eos_token}"
    )
    return {'text': text}

try:
    # Load dataset from your JSONL file
    dataset = load_dataset('json', data_files=DATASET_PATH, split="train")

    # Apply the formatting function. num_proc=1 is used for stable processing on M1.
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=['instruction', 'response'],
        num_proc=1 # THIS PREVENTS THE NAMEERROR
    )

    print(f"✅ Dataset loaded and formatted. Total examples: {len(dataset)}")
    
except Exception as e:
    print(f"❌ Error loading and processing dataset '{DATASET_PATH}': {e}")
    print("Please ensure the file exists and the structure is correct.")
    exit(1)


# --- 4. LoRA Configuration (PEFT) ---
LORA_TARGET_MODULES = [
    "q_proj", 
    "k_proj", 
    "v_proj", 
    "o_proj", 
    "gate_proj", 
    "up_proj", 
    "down_proj"
]

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model.enable_input_require_grads(True) # Required for certain LoRA/MPS setups
model = get_peft_model(model, lora_config)
print("\n--- LoRA Model Setup ---")
model.print_trainable_parameters()


# --- 5. Training Arguments (Tuning for Mac M1) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                     
    per_device_train_batch_size=1,          # CRITICAL: Batch size 1 for memory
    gradient_accumulation_steps=4,          # Effective batch size of 4
    optim="adamw_torch",                    
    learning_rate=2e-5,                     
    logging_steps=10,                       
    save_strategy="epoch",                  
    fp16=False,                             # Use float32 on MPS
    report_to="none",                       
    overwrite_output_dir=True,
    disable_tqdm=False,                     
)

# --- 6. Initialize and Run SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    max_seq_length=1024, 
    dataset_text_field="text",
    packing=False, 
)

print("\n--- Starting Fine-Tuning ---\n")
trainer.train()

# --- 7. Save the LoRA Adapter Weights ---
adapter_output_path = os.path.join(OUTPUT_DIR, LORA_ADAPTER_NAME)
trainer.model.save_pretrained(adapter_output_path)
print(f"\n✅ Fine-Tuning Complete!")
print(f"Adapter weights saved to: {adapter_output_path}")

# --- Next Steps ---
print("\nNext: Load the adapter and test performance on unseen headlines.")