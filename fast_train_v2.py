from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading GSM8K...")
gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split="train[:2000]"
)

print("Streaming OpenMath subset...")

openmath_stream = load_dataset(
    "nvidia/OpenMathReasoning",
    split="cot",
    streaming=True
)

openmath_list = []

for i, sample in enumerate(openmath_stream):
    if i >= 2000:
        break
    openmath_list.append(sample)

openmath = Dataset.from_list(openmath_list)

def format_gsm8k(example):
    text = f"""### Instruction:
Solve the math problem step by step.

### Problem:
{example['question']}

### Solution:
{example['answer']}
"""
    return {"text": text}

def format_openmath(example):

    question = example.get("problem", "")
    answer = example.get("generated_solution", "")

    text = f"""### Instruction:
Solve the math problem step by step.

### Problem:
{question}

### Solution:
{answer}
"""
    return {"text": text}

print("Formatting datasets...")

gsm8k = gsm8k.map(format_gsm8k)
openmath = openmath.map(format_openmath)

dataset = concatenate_datasets([gsm8k, openmath])

print("Tokenizing dataset...")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize)

print("Applying LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

print("Starting training...")

trainer.train()

print("Saving model...")

model.save_pretrained("./math_lora_model")
tokenizer.save_pretrained("./math_lora_model")

print("Training complete!")
