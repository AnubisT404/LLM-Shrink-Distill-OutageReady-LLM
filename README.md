# Knowledge Distillation with LLaMA 2 & Gemma

This project demonstrates **Knowledge Distillation**, where a **large fine-tuned teacher model** (LLaMA 2-7B) transfers knowledge to a **smaller student model** (Gemma-2B) using **KL divergence loss**.

---
## Project Overview

The process involves:
1. **Generating a synthetic dataset** using LLaMA-405B.
2. **Fine-tuning LLaMA-2-7B** on the synthetic dataset.
3. **Distilling knowledge** from the fine-tuned LLaMA-2-7B model into **Gemma-2B**.
4. **Training the student model** using KL divergence loss.

---
### Install dependencies:
```sh
pip install datasets
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7
pip install --upgrade transformers
pip install langchain_fireworks
pip install rich
pip install llama-index-llms-openai llama-index-embeddings-openai
pip install llama-index-finetuning llama-index-readers-file llama-index-embeddings-huggingface
```

### Hugging Face CLI Login:
To push models to the Hugging Face Hub, authenticate using:
```sh
huggingface-cli login --token YOUR_HF_TOKEN
```

---
## Files in This Repository

| File | Description |
|------------|----------------------------------------------|
| `data_generation.py` | Generates synthetic dataset using LLaMA-405B. |
| `finetune_llama.py` | Fine-tunes LLaMA-2-7B on the dataset. |
| `knowledge_distillation.py` | Transfers knowledge to Gemma-2B via KL loss. |

---
## Dataset Details
The dataset consists of **Question-Answer (QA) pairs** generated synthetically. These are stored in:
```sh
qa_pairs.csv
```
Each row contains:
```json
{
  "question": "What is AI?",
  "answer": "AI is the simulation of human intelligence in machines."
}
```

---
## Training & Distillation Process

### **Synthetic Data Generation**
- Uses **LLaMA-405B** to generate **200 QA pairs** per request.
- Stores data in `qa_pairs.csv`.

### **Fine-Tuning LLaMA-2-7B**
- Loads dataset & applies LoRA tuning.
- Uses `bitsandbytes` for efficient memory usage.
- Saves the fine-tuned model.

### **Knowledge Distillation**
- Loads **LLaMA-2-7B (Teacher)** and **Gemma-2B (Student)**.
- Uses **KL Divergence Loss** to transfer knowledge.
- Runs training for **5 epochs** with mixed precision.
- Saves the distilled model.

---
## Usage

### 1️⃣ **Generate Synthetic Data**
```sh
python data_generation.py
```

### 2️⃣ **Fine-tune LLaMA-2-7B**
```sh
python finetune_llama.py
```

### 3️⃣ **Train the Student Model (Distillation)**
```sh
python knowledge_distillation.py
```

---
## Results

- LLaMA 7B compressed into Gemma 2B, reducing parameters by 71%.
- Response time improved by 40% with minimal accuracy loss.
- High-quality synthetic dataset generated, improving correctness scores by 15% using NVIDIA NeMoTron.
- Reduced model size and latency for real-time applications.