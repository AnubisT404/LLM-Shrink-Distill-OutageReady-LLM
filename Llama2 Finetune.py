import os
import gc
import torch
import locale
import pandas as pd
from google.colab import files
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Define model paths and output directory
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
NEW_MODEL = "Llama-2-7b-chat-finetune"
OUTPUT_DIR = "./results"

# Upload and load dataset
uploaded = files.upload()
dataset = pd.read_csv("qa_pairs.csv").drop(columns=["Unnamed: 0"], axis=1)

dataset['text'] = 'Question:\n ' + dataset['question'] + '\n\nAnswer:\n ' + dataset['answer']
dataset.drop(columns=['question', 'answer'], inplace=True)
train = Dataset.from_pandas(dataset.iloc[:281, :])

# Configure 4-bit Quantization
compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load Pretrained Model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure LoRA Fine-tuning
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Train the Model
trainer.train()

# Save Fine-tuned Model
trainer.model.save_pretrained(NEW_MODEL)

# Clean VRAM
del model, trainer
gc.collect()

# Merge LoRA Weights with Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model = PeftModel.from_pretrained(base_model, NEW_MODEL)
model = model.merge_and_unload()

# Reload Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Push to Hugging Face Hub
locale.getpreferredencoding = lambda: "UTF-8"
#!huggingface-cli login --token "..."
model.push_to_hub("./Llama-2-7b-chat-finetune", check_pr=True)
tokenizer.push_to_hub("./Llama-2-7b-chat-finetune", check_pr=True)
