import ollama
import json
import os
from tqdm import tqdm # Import tqdm for a nice progress bar

# --- Configuration ---
OLLAMA_MODEL_NAME = "nemotron-mini:4b-instruct-q4_K_M" 
OUTPUT_FILE = "financial_synthetic_data.jsonl"
# Target 300 examples total for a quick proof-of-concept. Increase this for real training!
NUM_EXAMPLES_PER_SEED = 50 
# Seed prompts to ensure diversity and balanced sentiment in the dataset
SEED_PROMPTS = [
    {"focus": "Negative Earnings", "prompt": "Generate a headline and analysis focused on a major company missing quarterly earnings projections."},
    {"focus": "Positive Product Launch", "prompt": "Generate a headline and analysis focused on a successful product launch with high customer adoption."},
    {"focus": "Neutral Regulatory", "prompt": "Generate a headline and analysis focused on a minor, expected regulatory change with no immediate impact."},
    {"focus": "Negative Legal/Fine", "prompt": "Generate a headline and analysis focused on a major company facing a significant legal fine or penalty."},
    {"focus": "Positive Acquisition", "prompt": "Generate a headline and analysis focused on a strategic acquisition that boosts future growth outlook."},
    {"focus": "Neutral CEO Change", "prompt": "Generate a headline and analysis focused on a routine CEO retirement announcement."}
]

TOTAL_TARGET = len(SEED_PROMPTS) * NUM_EXAMPLES_PER_SEED

# Define the instruction template for the model
INSTRUCTION_TEMPLATE = (
    "As an expert financial analyst, generate a **brief, realistic news headline** "
    "and a corresponding **detailed JSON analysis** for a major company. "
    "The analysis must strictly use the following keys: `company_ticker`, "
    "`sentiment` ('Positive', 'Negative', or 'Neutral'), "
    "`event_type` (e.g., 'Product Launch', 'Regulatory Ruling', 'Quarterly Earnings'), and "
    "`expected_impact` ('Stock Up', 'Stock Down', or 'No Change'). "
    "Ensure the entire output is formatted cleanly with the headline first, followed by the JSON object."
)


# --- 1. Ollama Connection Check ---
try:
    client = ollama.Client()
    # Check if the model exists locally
    client.show(OLLAMA_MODEL_NAME)
    print("="*50)
    print(f"✅ Connected to Ollama. Using model: {OLLAMA_MODEL_NAME}")
    print(f"🎯 Target: {TOTAL_TARGET} synthetic data points.")
    print("="*50)
except Exception as e:
    print(f"❌ Error connecting to Ollama or finding model '{OLLAMA_MODEL_NAME}': {e}")
    print("   Please ensure the Ollama app is running and the model name is correct.")
    exit(1)


# --- 2. Helper Function for Generation ---
def generate_data_point(seed_prompt, max_tokens=250):
    """Generates a synthetic data point using the Ollama API."""
    
    # Combine the seed prompt with the instruction template
    full_prompt = f"{seed_prompt}\n\n{INSTRUCTION_TEMPLATE}"
    
    # Ollama uses the chat format internally, which respects the instruction better
    messages = [
        {'role': 'system', 'content': "You are a specialized AI designed to generate synthetic financial news and structured analysis."},
        {'role': 'user', 'content': full_prompt}
    ]

    response = client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=messages,
        options={
            'num_predict': max_tokens,
            'temperature': 0.8, # Use high temperature for diverse output
            'top_p': 0.95
        }
    )
    
    return response['message']['content'].strip()

# --- 3. Data Generation and Parsing Loop ---

total_generated = 0
print(f"\nStarting generation. Writing results to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for seed in SEED_PROMPTS:
        print(f"\n--- Generating Batch: {seed['focus']} ---")
        
        # Use tqdm for a progress bar
        for i in tqdm(range(NUM_EXAMPLES_PER_SEED)):
            try:
                raw_response = generate_data_point(seed['prompt'])
                
                # --- Parsing Logic ---
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    json_text = raw_response[json_start : json_end + 1]
                    headline_text = raw_response[:json_start].strip()

                    # Try to load the JSON to validate and clean the data point
                    data_json = json.loads(json_text)
                    
                    # Ensure the JSON has the required keys before saving
                    required_keys = ['company_ticker', 'sentiment', 'event_type', 'expected_impact']
                    if all(key in data_json for key in required_keys):
                        
                        # Create the final training data format (Instruction + Response)
                        # We use the headline as the instruction input for the fine-tuned model
                        training_example = {
                            "instruction": headline_text,
                            "response": json.dumps(data_json) # Save the clean, valid JSON
                        }
                        
                        f.write(json.dumps(training_example) + '\n')
                        total_generated += 1
                        
                    # else: skip this iteration due to bad keys
                # else: skip this iteration due to no JSON found
                
            except json.JSONDecodeError:
                pass # Skip if JSON is malformed
            except Exception as e:
                pass # Skip on other errors

print("\n" + "="*50)
print(f"✅ Data Generation Complete!")
print(f"📊 Total valid examples saved: {total_generated} out of a possible {TOTAL_TARGET}")
print(f"📝 Output File: {OUTPUT_FILE}")
print("="*50)