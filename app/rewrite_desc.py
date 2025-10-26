import os
import requests
from dotenv import load_dotenv

load_dotenv()

INPUT_DIR = "docs_txt"
OUTPUT_DIR = "docs_txt_natural"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b"

def generate_natural_description(text):
    prompt_text = (
        "Rewrite the following function documentation into a single natural-language paragraph. "
        "Include what the function does, how to use it, any limitations or dependencies. "
        "Do not include headings or tables, just natural flowing text:\n\n"
        f"{text}\n\nNatural paragraph:"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2,
        "stream": False  # Important: disable streaming for simpler parsing
    }
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        # The response structure is: {"message": {"content": "..."}, ...}
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"Failed to get completion: {e}")
        return ""


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    natural_text = generate_natural_description(content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(natural_text)

def process_folder(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            relative_path = os.path.relpath(root, input_dir)
            target_dir = os.path.join(output_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            input_path = os.path.join(root, file)
            output_path = os.path.join(target_dir, file)

            print(f"Processing {input_path}")
            process_file(input_path, output_path)

if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print("All files processed. Natural descriptions saved to", OUTPUT_DIR)
