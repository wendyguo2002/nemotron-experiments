import ollama

print("="*50)
print("NEMOTRON 4B via Ollama API")
print("="*50)

# === Configuration ===
OLLAMA_MODEL_NAME = "nemotron-mini:4b-instruct-q4_K_M"
TEST_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms.",
    "What is supervised learning?"
]

print(f"\n🔄 Connecting to Ollama and using {OLLAMA_MODEL_NAME}...")

# === HELPER FUNCTION ===
def ask_ollama(prompt, max_tokens=200, system_prompt=None):
    """Generate response from Ollama API"""
    
    # Ollama uses a list of messages for conversation
    messages = []
    
    # Nemotron-Mini-4B uses a specific template which Ollama handles automatically
    # when using the 'chat' endpoint and system/user roles.
    
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
        
    messages.append({'role': 'user', 'content': prompt})
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=messages,
            options={
                'num_predict': max_tokens, # max_new_tokens
                'temperature': 0.7,
                'top_p': 0.95
            }
        )
        # The response is a dictionary; extract the content
        return response['message']['content'].strip()
    
    except Exception as e:
        return f"❌ Ollama Error: {e}"

# === LEVEL 1: BASIC INFERENCE ===
print("\n" + "="*50)
print("LEVEL 1: BASIC INFERENCE")
print("="*50)

# Optional: You can pre-define a system prompt for the entire conversation
# system_instruction = "You are a helpful and concise AI assistant."

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n[{i}/{len(TEST_PROMPTS)}] 📝 {prompt}")
    
    response = ask_ollama(prompt, max_tokens=150)
    print(f"💬 {response}\n")

print("\n" + "="*50)
print("✅ COMPLETE!")
print("="*50)
print(f"🤖 Model: {OLLAMA_MODEL_NAME} (via Ollama)")
print("⚙️  Deployment: Local, optimized via GGUF/Metal (MPS)")