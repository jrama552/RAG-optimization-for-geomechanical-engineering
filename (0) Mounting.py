from google.colab import drive
import json
import re

drive.mount('/content/drive')

json_path = '/content/drive/MyDrive/file.json'
with open(json_path, 'r') as f:
    data = json.load(f) 

true_code = [json.dumps(d['files']) for d in data]
true_code = true_code[120:]

with open("drive/MyDrive/LLM project/file", "r") as file:
    base_code = json.load(file)

with open("drive/MyDrive/LLM project/code/file", "r") as file:
    few_shot_code = json.load(file)
