import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW
)

# Constants
STUDENT_MODEL_ID = "google/gemma-2-2b-it"
TEACHER_MODEL_ID = "./Llama-2-7b-chat-finetune"
DATA_PATH = "qa_pairs.csv"
EPOCHS = 5
LEARNING_RATE = 5e-5


def load_dataset(filepath):
    """Loads and processes the dataset from a CSV file."""
    dataset = pd.read_csv(filepath).drop(columns=["Unnamed: 0"], errors='ignore')
    dataset['text'] = 'Question:\n ' + dataset['question'] + '\n\nAnswer:\n ' + dataset['answer']
    dataset.drop(columns=['question', 'answer'], axis=1, inplace=True)
    return dataset


def load_models():
    """Loads the teacher and student models."""
    student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID)
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    
    # Adjust vocabulary size if needed
    if student_model.config.vocab_size != teacher_model.config.vocab_size:
        student_model.resize_token_embeddings(teacher_model.config.vocab_size)
    
    return student_model, teacher_model, tokenizer


def distillation_loss(student_logits, teacher_logits):
    """Computes the KL Divergence loss between teacher and student model outputs."""
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    student_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    return loss_fct(student_probs, teacher_probs)


def train_student_model(student_model, teacher_model, tokenizer, dataset, num_epochs, lr):
    """Trains the student model using knowledge distillation."""
    optimizer = AdamW(student_model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0
        
        for idx in tqdm(range(len(dataset[:281])), desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = tokenizer(dataset.iloc[idx]['text'], return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(student_model.device) for key, value in inputs.items()}
            optimizer.zero_grad()
            
            # Teacher Model Prediction (No Gradient Calculation)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Student Model Training (Mixed Precision)
            with torch.cuda.amp.autocast():
                student_outputs = student_model(**inputs)
                student_logits = student_outputs.logits
                loss = distillation_loss(student_logits, teacher_logits)
            
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset(DATA_PATH)
    
    print("Loading models...")
    student_model, teacher_model, tokenizer = load_models()
    
    print("Starting knowledge distillation training...")
    train_student_model(student_model, teacher_model, tokenizer, dataset, EPOCHS, LEARNING_RATE)

