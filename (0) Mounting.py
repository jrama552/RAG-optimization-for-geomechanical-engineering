import json
import re
import os

json_path = 'data/synthetic_prompts.json'
with open(json_path, 'r') as f:
    data = json.load(f)

synthetic_code = [json.dumps(d['files']) for d in data]
synthetic_code = synthetic_code[120:]

with open("data/base_examples.json", "r") as file:
    base_code = json.load(file)

with open("data/few_shot_examples.json", "r") as file:
    few_shot_code = json.load(file)
