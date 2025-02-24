import os
import json
import pandas as pd
from tqdm.notebook import tqdm
from rich import print
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_fireworks import ChatFireworks

# Initialize Fireworks LLaMA model
llama_405B = ChatFireworks(
    model="././models/llama-v3p1-405b-instruct",
    temperature=0.7,
    api_key=".."
)

# Define prompt template for generating QA pairs
qa_cot_prompt = """
You are a highly skilled expert with deep knowledge of various topics.
Your task is to prepare thoughtful and informative questions and corresponding answers on the given TOPIC.
You will generate exactly "n" question-answer pairs.

Respond only in the following JSON format:
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."},
  ...
]
"""

# Define prompt chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_cot_prompt),
    ("human", "Your TOPIC to analyze: {topic}. You need to generate {n} question-answer pairs.")
])

chain = qa_prompt | llama_405B | JsonOutputParser()

# Generate synthetic dataset
responses = []
for _ in range(10):
    try:
        response = chain.invoke({"topic": "Artificial Intelligence", "n": 200})
        responses.append(response)
    except Exception as e:
        print(f"Error encountered: {e}")
        break

# Process responses into DataFrame
questions, answers = [], []
for response_item in responses:
    for qa_pair in response_item:
        try:
            questions.append(qa_pair["question"])
            answers.append(qa_pair["answer"])
        except KeyError:
            answers.append("None")  # Handle missing data

# Filter out invalid QA pairs
valid_qa_pairs = [
    {"question": q, "answer": a} for q, a in zip(questions, answers) if a != "None"
]

df = pd.DataFrame(valid_qa_pairs)
df.to_csv("qa_pairs.csv", index=False)
print("Generated around 350 rows of QA data and saved to qa_pairs.csv")

# Prepare synthetic data for fine-tuning
question_response_pair_list = [
    {"question": row['question'], "responses": {"response_a": {"response": row['answer']}}}
    for _, row in df.iterrows()
]

# Save data to JSONL format
with open('synthetic_data.jsonl', 'w') as f:
    for item in question_response_pair_list:
        f.write(json.dumps(item) + '\n')