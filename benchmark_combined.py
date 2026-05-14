from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time
import re
import gc

GSM8K_SAMPLES = 200
OPENMATH_SAMPLES = 100

models = [
    {
        "name": "Qwen2.5-Math-1.5B",
        "hf": "Qwen/Qwen2.5-Math-1.5B",
        "lora": None
    },
    {
        "name": "Qwen2.5-Math-1.5B-LoRA",
        "hf": "Qwen/Qwen2.5-Math-1.5B",
        "lora": "./math_lora_model"
    },
    {
        "name": "Qwen3-1.7B",
        "hf": "Qwen/Qwen3-1.7B",
        "lora": None
    },
    {
        "name": "Qwen3.5-2B-Base",
        "hf": "Qwen/Qwen3.5-2B-Base",
        "lora": None
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "lora": None
    }
]

print("Loading GSM8K...")
gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split=f"test[:{GSM8K_SAMPLES}]"
)

print("Streaming OpenMathReasoning...")

openmath_stream = load_dataset(
    "nvidia/OpenMathReasoning",
    split="cot",
    streaming=True
)

openmath_list = []

for i, sample in enumerate(openmath_stream):

    if i >= OPENMATH_SAMPLES:
        break

    openmath_list.append(sample)

openmath = Dataset.from_list(openmath_list)

results = []

def extract_number(text):

    matches = re.findall(
        r"[-+]?\d*\.\d+|\d+",
        text.replace(",", "")
    )

    if matches:
        return matches[-1]

    return None

def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    dataset_name,
    question_key,
    answer_key
):

    correct = 0
    total_time = 0
    total = len(dataset)

    print(f"\nEvaluating on {dataset_name}...")

    for idx, sample in enumerate(dataset):

        question = sample.get(question_key, "")

        answer = sample.get(answer_key, "")

        gt = extract_number(str(answer))

        prompt = f"""
Solve this math problem step by step.

Question:
{question}

Final Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        start = time.time()

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False
            )

        end = time.time()

        total_time += (end - start)

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        pred = extract_number(response)

        if pred == gt:
            correct += 1

        print(
            f"[{idx+1}/{total}] "
            f"GT={gt} | "
            f"PRED={pred}"
        )

    accuracy = (correct / total) * 100
    avg_time = total_time / total

    return accuracy, avg_time

for model_info in models:

    print("\n===================================================")
    print(f"Evaluating Model: {model_info['name']}")
    print("===================================================")

    tokenizer = AutoTokenizer.from_pretrained(
        model_info["hf"]
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_info["hf"],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if model_info["lora"] is not None:

        print("Loading LoRA adapter...")

        model = PeftModel.from_pretrained(
            model,
            model_info["lora"]
        )

    model.eval()

    gsm8k_acc, gsm8k_time = evaluate_dataset(
        model,
        tokenizer,
        gsm8k,
        "GSM8K",
        "question",
        "answer"
    )

    openmath_acc, openmath_time = evaluate_dataset(
        model,
        tokenizer,
        openmath,
        "OpenMathReasoning",
        "problem",
        "generated_solution"
    )

    combined_acc = (gsm8k_acc + openmath_acc) / 2
    combined_time = (gsm8k_time + openmath_time) / 2

    results.append({
        "Model": model_info["name"],
        "GSM8K": gsm8k_acc,
        "OpenMath": openmath_acc,
        "Combined": combined_acc,
        "AvgTime": combined_time
    })

    print("\nMODEL RESULTS")
    print(f"GSM8K Accuracy: {gsm8k_acc:.2f}%")
    print(f"OpenMath Accuracy: {openmath_acc:.2f}%")
    print(f"Combined Accuracy: {combined_acc:.2f}%")
    print(f"Average Generation Time: {combined_time:.2f} sec")

    del model
    gc.collect()
    torch.cuda.empty_cache()

print("\n===================================================")
print("FINAL BENCHMARK RESULTS")
print("===================================================")

for r in results:

    print(
        f"{r['Model']} | "
        f"GSM8K: {r['GSM8K']:.2f}% | "
        f"OpenMath: {r['OpenMath']:.2f}% | "
        f"Combined: {r['Combined']:.2f}% | "
        f"Avg Time: {r['AvgTime']:.2f} sec"
    )
