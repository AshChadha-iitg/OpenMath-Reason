# OpenMath-Reason

## Optimizing Mathematical Reasoning in Tiny Language Models with LoRA

OpenMath-Reason is a lightweight mathematical reasoning Tiny Language Model (TLM) developed using LoRA-based fine-tuning on AMD Instinct MI300X GPUs with ROCm-enabled workflows.

This project benchmarks compact reasoning-oriented language models across GSM8K and OpenMathReasoning datasets and demonstrates that parameter-efficient fine-tuning can significantly improve mathematical reasoning performance in small models.

---

# Features

- LoRA-based mathematical reasoning fine-tuning
- ROCm + AMD MI300X optimized workflows
- GSM8K benchmark evaluation
- OpenMathReasoning evaluation
- Tiny Language Model benchmarking
- Inference latency vs intelligence analysis
- Lightweight mathematical reasoning adaptation

---

# Evaluated Models

| Model | Parameters |
|---|---|
| Qwen2.5-Math-1.5B | 1.5B |
| OpenMath-Reason | 1.5B |
| Qwen3-1.7B | 1.7B |
| Qwen3.5-2B-Base | 2B |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B |

---

# Hardware

Experiments were conducted on:

- AMD Instinct MI300X
- ROCm 7.0
- PyTorch 2.6
- 192GB HBM3 GPU Memory

---

# Datasets

## GSM8K
Grade-school mathematical reasoning benchmark.

## OpenMathReasoning
Chain-of-thought mathematical reasoning dataset.

---

# Benchmark Results

| Model | GSM8K (%) | OpenMath (%) | Combined (%) | Avg Time (s) |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B | 14.50 | 26.00 | 20.25 | 1.87 |
| OpenMath-Reason | **29.00** | **26.00** | **27.50** | 3.18 |
| Qwen3-1.7B | 6.00 | 17.00 | 11.50 | 2.34 |
| Qwen3.5-2B-Base | 20.50 | 24.00 | 22.25 | 2.86 |
| DeepSeek-R1-Distill-Qwen-1.5B | 5.50 | 14.00 | 9.75 | 1.95 |

---

# Training

The model was fine-tuned using:

- LoRA (Low-Rank Adaptation)
- PEFT
- Hugging Face Transformers
- ROCm-enabled PyTorch

Training was conducted for 2 epochs using reduced subsets of GSM8K and OpenMathReasoning for rapid experimentation.

---

# Running Evaluation

Example:

```bash
python benchmark_combined.py
```

---

# Key Findings

- OpenMath-Reason achieved the highest combined benchmark accuracy.
- LoRA adaptation significantly improved GSM8K performance.
- ROCm workflows proved effective for compact reasoning-oriented AI experimentation.
- Tiny Language Models can achieve meaningful mathematical reasoning improvements without frontier-scale infrastructure.

---

# Technologies Used

- ROCm
- AMD Instinct MI300X
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA
- Python
- GSM8K
- OpenMathReasoning

---

# Future Work

- Larger-scale benchmark evaluation
- MATH and AIME integration
- Multi-stage reasoning fine-tuning
- Self-consistency decoding
- Distributed ROCm training

---

# License

Apache 2.0
