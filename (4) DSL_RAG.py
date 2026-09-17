from sentence_transformers import SentenceTransformer

# init sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
train_data = data[0:120]
test_data = data[120:]
# making embeddings for prompt
train_name_prompts = [d['project'] + ": /n" + d['prompt'] for d in train_data]
train_codes = [d['files'] for d in train_data]
train_embeddings = model.encode(train_name_prompts)
test_name_prompts = [d['project'] + ": /n" + d['prompt'] for d in test_data]
test_codes = [d['files'] for d in test_data]
test_embeddings = model.encode(test_name_prompts)

print("Embeddings created successfully:")

from scipy.spatial.distance import cosine
from openai import OpenAI

client = OpenAI(api_key="iLOVELOVELOVEpython")
outputs = []

for i in range (len(test_data)):
  test_embedding = test_embeddings[i]
  similarities =  [1 - cosine(test_embedding, embedding) for embedding in train_embeddings]

  closest_index = similarities.index(max(similarities))

  closest_np = train_name_prompts[closest_index]
  closest_code = ''.join(f"{k}:{v};" for d in train_codes[closest_index] for k, v in d.items())
  if (len(closest_code)>20000):
    closest_code = closest_code[:20000]
    closest_code += "..."

  info=data[closest_index]

  text = "Here is a relevant example of a prompt and good real-ESSI code"
  text+=closest_np + "/n" + closest_code
  text += " Learning from that example, your job is to implement code for this and only output those code file(s)." + test_name_prompts[i]

  response = client.responses.create(
      model="gpt-4.1",
      input=text
  )
  outputs.append(response.output_text)
