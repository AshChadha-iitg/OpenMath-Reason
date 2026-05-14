from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time

BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
LORA_MODEL = "./math_lora_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_MODEL)

prompt = """
Solve this math problem step by step:

If a train travels 120 km in 2 hours, what is its average speed?
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")

start = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7
)

end = time.time()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====\n")
print(response)

print("\n===== METRICS =====")
print(f"Generation Time: {end-start:.2f} sec")
