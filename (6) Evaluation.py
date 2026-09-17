from openai import OpenAI
import os
import json
import time
from google.colab import drive

client = OpenAI(api_key="hmmIhaventplayedtheWordletoday")

drive.mount('/content/drive')
base_path = "/content/drive/MyDrive/file"  # Change to your real base path

with open(os.path.join(base_path, "file.json"), "r") as f:
  prompts_data = json.load(f)

# --------------------------------------------------------------------------------------------------------------

prompts_data_train = prompts_data[0:120]
prompts_data_test = prompts_data[120:]
prompts_test = [d['prompt'] for d in prompts_data_test]
true_code = [d['files'] for d in prompts_data_test]

# --------------------------------------------------------------------------------------------------------------

with open("drive/MyDrive/LLM project/code/file.json", "r") as file:
    base_code = json.load(file)

with open("drive/MyDrive/LLM project/code/file.json", "r") as file:
    few_shot_code = json.load(file)

true_code_strings = []
for project in true_code:
    project_string = json.dumps(project)
    true_code_strings.append(project_string)

# --------------------------------------------------------------------------------------------------------------


from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# --------------------------------------------------------------------------------------------------------------

import torch

def get_code_embeddings(code_strings, tokenizer, model, max_length=512):
    """
    Generates CodeBERT embeddings for a list of code strings.

    Args:
        code_strings: A list of code strings.
        tokenizer: The loaded CodeBERT tokenizer.
        model: The loaded CodeBERT model.
        max_length: The maximum sequence length for tokenization.

    Returns:
        A list of numpy arrays, where each array is the embedding for a code string.
    """
    embeddings = []
    for code in code_strings:
        # Tokenize the code string
        encoded_input = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        # Move tensors to the same device as the model
        input_ids = encoded_input['input_ids'].to(model.device)
        attention_mask = encoded_input['attention_mask'].to(model.device)

        # Get embeddings from the model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Use the mean of the token embeddings as the code embedding
        # You could also use outputs.last_hidden_state[:, 0, :] for the [CLS] token embedding
        code_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(code_embedding)

    return embeddings
# --------------------------------------------------------------------------------------------------------------


true_code_embeddings = get_code_embeddings(true_code_strings, tokenizer, model)
base_code_embeddings = get_code_embeddings(base_code, tokenizer, model)
few_shot_code_embeddings = get_code_embeddings(few_shot_code, tokenizer, model)

# --------------------------------------------------------------------------------------------------------------


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

true_code_embeddings_np = np.array(true_code_embeddings)
base_code_embeddings_np = np.array(base_code_embeddings)
few_shot_code_embeddings_np = np.array(few_shot_code_embeddings)

base_similarity_matrix = cosine_similarity(true_code_embeddings_np, base_code_embeddings_np)
few_shot_similarity_matrix = cosine_similarity(true_code_embeddings_np, few_shot_code_embeddings_np)

# --------------------------------------------------------------------------------------------------------------
import numpy as np

base_pairwise_similarity_scores = np.diag(base_similarity_matrix)

base_average_similarity = np.mean(base_pairwise_similarity_scores)

base_min_similarity = np.min(base_pairwise_similarity_scores)

base_max_similarity = np.max(base_pairwise_similarity_scores)

# results
print("Pairwise Similarity Scores:")
for i, score in enumerate(base_pairwise_similarity_scores):
    print(f"  Pair {i+1}: {score:.4f}")

print(f"\nOverall Average Similarity: {base_average_similarity:.4f}")
print(f"Minimum Similarity: {base_min_similarity:.4f}")
print(f"Maximum Similarity: {base_max_similarity:.4f}")

# --------------------------------------------------------------------------------------------------------------


few_shot_pairwise_similarity_scores = np.diag(few_shot_similarity_matrix)

few_shot_average_similarity = np.mean(few_shot_pairwise_similarity_scores)

few_shot_min_similarity = np.min(few_shot_pairwise_similarity_scores)

few_shot_max_similarity = np.max(few_shot_pairwise_similarity_scores)


#  results (but for few shot)
print("Pairwise Similarity Scores:")
for i, score in enumerate(few_shot_pairwise_similarity_scores):
   print(f"  Pair {i+1}: {score:.4f}")


print(f"\nOverall Average Similarity: {few_shot_average_similarity:.4f}")
print(f"Minimum Similarity: {few_shot_min_similarity:.4f}")
print(f"Maximum Similarity: {few_shot_max_similarity:.4f}")

# --------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

base_pairwise_similarity_scores = np.diag(base_similarity_matrix)
few_shot_pairwise_similarity_scores = np.diag(few_shot_similarity_matrix)

x = np.arange(1, len(base_pairwise_similarity_scores) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x, base_pairwise_similarity_scores, marker='o', label='Base Similarity')
plt.plot(x, few_shot_pairwise_similarity_scores, marker='s', label='Few-shot Similarity')

plt.title('Pairwise Similarity Scores Comparison')
plt.xlabel('Pair Index')
plt.ylabel('Similarity Score')
plt.ylim(0.8, 1.05)  # bc normalized scores
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()

