import os
import json
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

base_path = "data/synthetic_projects"

results = []
MAX_LENGTH = 10000

for folder_entry in os.scandir(base_path):
    if not folder_entry.is_dir():
        continue

    project_name = folder_entry.name
    print(f"Processing project: {project_name}")

    files_data = []
    for file_entry in os.scandir(folder_entry.path):
        if file_entry.is_file():
            with open(file_entry.path, "r", encoding="utf-8") as f:
                content = f.read()
            files_data.append({"name": file_entry.name, "content": content})

    if not files_data:
        continue

    all_contents = "\n\n".join(
        f"# File: {f['name']}\n{f['content']}" for f in files_data
    )

    if len(all_contents) <= MAX_LENGTH:
        file_contents = all_contents
        prompt_intro = (
            f"You are provided with the following synthetic DSL files for a project called '{project_name}':"
        )
    else:
        main_file = next(
            (f for f in files_data if f["name"].lower().startswith("main")), None
        )
        if main_file:
            file_contents = f"# File: {main_file['name']}\n{main_file['content']}"
            prompt_intro = (
                f"The total length of files is too large. "
                f"Only the main synthetic file '{main_file['name']}' for project '{project_name}' is provided below:"
            )
        else:
            continue

    user_prompt = (
        f"{prompt_intro}\n"
        f"{file_contents}\n\n"
        """
        Based on the provided synthetic file(s), generate a detailed natural language prompt
        that a developer could use to reproduce the same functionality using a language model.
        Do not include code. Output only the prompt text.
        """
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": user_prompt}]
    )

    assistant_reply = response.choices[0].message.content.strip()

    results.append({
        "project": project_name,
        "prompt": assistant_reply,
        "files": files_data
    })

with open("data/synthetic_prompt_code_pairs.json", "w") as f:
    json.dump(results, f, indent=2)
