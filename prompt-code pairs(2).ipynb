from openai import OpenAI
import os
import json
import time

client = OpenAI(api_key="[iLOVEpython]")

base_path = "/content/drive/MyDrive/LLM project/code/fei_codes (1)"  # Change to your real base path

results = []
MAX_LENGTH = 10000  # Character limit for prompt content

for folder_entry in os.scandir(base_path):
    run = True
    if not folder_entry.is_dir():
        continue

    project_name = folder_entry.name
    print(f"Processing project: {project_name}")

    files_data = []
    for file_entry in os.scandir(folder_entry.path):
        if file_entry.is_file():
            try:
                with open(file_entry.path, "r", encoding="utf-8") as f:
                    content = f.read()
                files_data.append({
                    "name": file_entry.name,
                    "content": content
                })
            except Exception as e:
                print(f"Warning: Could not read file {file_entry.name}: {e}")

    if not files_data:
        print(f"Skipping project '{project_name}': No readable files found.")
        continue

    all_contents = "\n\n".join(f"# File: {f['name']}\n{f['content']}" for f in files_data)

    if len(all_contents) <= MAX_LENGTH:
        # Use all files
        file_contents = all_contents
        prompt_intro = (
            f"You are provided with the following FEI code files for a simulation project called '{project_name}':"
        )
    else:
        # Try to find the main file by name (case-insensitive)
        main_file = next((f for f in files_data if f["name"].lower().startswith("main")), None)

        if main_file:
            file_contents = f"# File: {main_file['name']}\n{main_file['content']}"
            prompt_intro = (
                f"The total length of files is too large. "
                f"Only the main file '{main_file['name']}' for the simulation project '{project_name}' is provided below:"
            )
            if (len(main_file['content']) > MAX_LENGTH):
                print(f"Warning: The main file for'{main_file['name']}' is too large to process.")
                run = False
        else:
           run = False
           print(f"Warning: No main file found for the simulation project '{project_name}'.")

    user_prompt = (
        f"{prompt_intro}\n"
        f"{file_contents}\n\n"
        """
        Based on the provided code file(s), generate a highly detailed and specific natural language prompt that a developer could use to accurately reproduce the same functionality using a language model.
        The prompt should:

        Clearly describe the purpose and functionality of the program

        Specify any libraries, packages, or frameworks used

        Describe the inputs, outputs, and processing logic in detail

        Mention any assumptions, constraints, or configurations

        Reflect the structure of the code (e.g., number of files, key functions, classes, or modules)
        Use clear, natural language phrasing as if explaining to a competent developer or AI model.
        Do not include the code itself. Output only the resulting prompt text. Do not use any special characters.
        """
    )

    if run:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": user_prompt}]
        )

        assistant_reply = response.choices[0].message.content.strip()
        print(f"Generated prompt for '{project_name}':\n{assistant_reply}\n")

        results.append({
            "project": project_name,
            "prompt": assistant_reply,
            "files": files_data
        })
