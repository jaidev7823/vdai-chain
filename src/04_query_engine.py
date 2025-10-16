import json
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# === Initialize models ===
embed_model = OllamaEmbedding(model_name="all-minilm")
llm = Ollama(model="mistral", request_timeout=12000)

# === Step 1: Break query into sub-actions ===
def decompose_query(query: str):
    system_prompt = """You are a Premiere Pro automation agent. 
Break the user request into clear step-by-step actions required to fulfill it.
Each action must represent a concrete operation like “find clip”, “apply effect”, “rename layer”, “adjust property”, etc.
Return as a numbered JSON array like:
[
  {"action": "find clip"},
  {"action": "apply glow effect"},
  {"action": "rename adjustment layer"}
]"""

    prompt = f"{system_prompt}\n\nUser request: {query}"
    result = llm.complete(prompt).text
    try:
        actions = json.loads(result.strip())
    except:
        actions = [{"action": line.strip()} for line in result.split("\n") if line.strip()]
    return actions


# === Step 4: Main Pipeline ===
def process_query(query: str):
    actions = decompose_query(query)
    full_plan = []

    for action in actions:
        print(f"\n🧩 Action: {action['action']}")
        full_plan.append(action)

    return full_plan


# === Example Usage ===
if __name__ == "__main__":
    user_query = "add red glow on this clip"
    plan = process_query(user_query)

    print("\n=== FINAL PLAN ===")
    print(json.dumps(plan, indent=2))
