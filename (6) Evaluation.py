from openai import OpenAI
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

client = OpenAI(api_key="YOUR_API_KEY")

with open("data/synthetic_prompt_code_pairs.json", "r") as f:
    prompts_data = json.load(f)

train_data = prompts_data[:120]
test_data = prompts_data[120:]

prompts_test = [d['prompt'] for d in test_data]
true_code = [d['files'] for d in test_data]

with open("data/base_examples.json", "r") as file:
    base_code = json.load(file)

with open("data/few_shot_examples.json", "r") as file:
    few_shot_code = json.load(file)

true_code_strings = [json.dumps(project) for project in true_code]

# CodeBERT
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_code_embeddings(code_strings, tokenizer, model, max_length=512):
    embeddings = []
    for code in code_strings:
        encoded_input = tokenizer(
            code,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )

        input_ids = encoded_input['input_ids'].to(model.device)
        attention_mask = encoded_input['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        code_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(code_embedding)

    return embeddings

true_code_embeddings = get_code_embeddings(true_code_strings, tokenizer, model)
base_code_embeddings = get_code_embeddings(base_code, tokenizer, model)
few_shot_code_embeddings = get_code_embeddings(few_shot_code, tokenizer, model)

true_np = np.array(true_code_embeddings)
base_np = np.array(base_code_embeddings)
few_np = np.array(few_shot_code_embeddings)

base_similarity_matrix = cosine_similarity(true_np, base_np)
few_similarity_matrix = cosine_similarity(true_np, few_np)

base_scores = np.diag(base_similarity_matrix)
few_scores = np.diag(few_similarity_matrix)

print("Base Similarity Scores:")
for i, score in enumerate(base_scores):
    print(f"Pair {i+1}: {score:.4f}")

print("\nFew-shot Similarity Scores:")
for i, score in enumerate(few_scores):
    print(f"Pair {i+1}: {score:.4f}")

x = np.arange(1, len(base_scores) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x, base_scores, marker='o', label='Base Similarity')
plt.plot(x, few_scores, marker='s', label='Few-shot Similarity')

plt.title('Synthetic Code Similarity Comparison')
plt.xlabel('Pair Index')
plt.ylabel('Similarity Score')
plt.ylim(0.8, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
