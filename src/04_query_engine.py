import json
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from pathlib import Path

# === Initialize models ===
embed_model = OllamaEmbedding(model_name="all-minilm")
llm = Ollama(model="mistral", request_timeout=12000)

# === Step 1: Break query into sub-actions ===
def decompose_query(query: str):
    prompt_file = Path("prompt/prompt.txt")
    system_prompt = prompt_file.read_text(encoding="utf-8")
    prompt = f"{system_prompt}\n\nUser request: {query}"
    result = llm.complete(prompt).text.strip()

    # quick sanitize to remove markdown or extra text
    cleaned = (
        result.replace("```json", "")
              .replace("```", "")
              .replace("Here is a step-by-step guide for the requested action in Premiere Pro:", "")
              .strip()
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: return each line as an action if JSON fails
        return [{"action": line.strip(), "description": ""} for line in cleaned.split("\n") if line.strip()]

# === Main Pipeline ===
if __name__ == "__main__":
    user_query = "i want to add glowing effect on the mobile in left the animation should be from top to making round on the mobile"
    plan = decompose_query(user_query)

    print("\n=== FINAL PLAN ===")
    print(json.dumps(plan, indent=2))

# === Function retrieval code (commented out for now) ===
# def retrieve_functions_for_action(action_text: str, top_k=5, confidence_threshold=0.75):
#     pass
#
# def explain_action_with_functions(action, candidates):
#     pass
#
# def process_query(query: str):
#     actions = decompose_query(query)
#     full_plan = []
#     for action in actions:
#         candidates = retrieve_functions_for_action(action["action"])
#         explanation = explain_action_with_functions(action, candidates)
#         full_plan.append({
#             "action": action["action"],
#             "candidates": candidates,
#             "decision": explanation,
#         })
#     return full_plan
