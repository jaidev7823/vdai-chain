import json
import sqlite3
import faiss
from pathlib import Path
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# === Paths ===
DB_PATH = Path("embeddings/metadata.db")
FAISS_DIR = Path("embeddings/faiss_indexes")

# === Load FAISS indexes ===
faiss_indexes = {
    "description": faiss.read_index(str(FAISS_DIR / "description.index")),
    "details": faiss.read_index(str(FAISS_DIR / "details.index")),
    "example": faiss.read_index(str(FAISS_DIR / "example.index")),
}

# === Connect DB ===
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# === Initialize models ===
embed_model = OllamaEmbedding(model_name="all-minilm")
llm = Ollama(model="mistral", request_timeout=12000)


# === Utility: get embedding ===
def embed(text: str):
    return embed_model.get_text_embedding(text).reshape(1, -1)


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
    result = llm.complete(prompt)
    try:
        actions = json.loads(result.strip())
    except:
        actions = [{"action": line.strip()} for line in result.split("\n") if line.strip()]
    return actions


# === Step 2: Retrieve top functions for each action ===
def retrieve_functions_for_action(action_text: str, top_k=5, confidence_threshold=0.75):
    query_vector = embed(action_text)
    results = []

    for field, index in faiss_indexes.items():
        distances, indices = index.search(query_vector, top_k)
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            confidence = float(1 / (1 + distances[0][i]))  # normalized inverse distance
            if confidence < confidence_threshold:
                continue

            cur.execute(f"SELECT * FROM metadata WHERE faiss_id_{field} = ?", (int(idx),))
            row = cur.fetchone()
            if not row:
                continue

            results.append({
                "field": field,
                "function": f"{row['object_name']}.{row['member_name']}",
                "confidence": round(confidence, 3),
                "desc": row["full_signature"],
                "section": row["section"],
            })
    return sorted(results, key=lambda x: -x["confidence"])[:top_k]


# === Step 3: LLM rationale generation ===
def explain_action_with_functions(action, candidates):
    func_list = "\n".join([f"- {c['function']} (conf: {c['confidence']})" for c in candidates])
    context = "\n".join([f"{c['function']} → {c['desc']}" for c in candidates])
    prompt = f"""You are an expert Premiere Pro API planner.
Given the action: "{action['action']}"
and these available functions:
{func_list}

Explain which function(s) are most suitable and why.
Use only provided context:
{context}

Respond in a compact structured JSON like:
{{
  "best_function": "...",
  "reason": "...",
  "alternatives": ["...", "..."]
}}"""

    response = llm.complete(prompt)
    try:
        return json.loads(response.strip())
    except:
        return {"best_function": None, "reason": response.strip(), "alternatives": []}


# === Step 4: Main Pipeline ===
def process_query(query: str):
    actions = decompose_query(query)
    full_plan = []

    for action in actions:
        print(f"\n🧩 Action: {action['action']}")
        candidates = retrieve_functions_for_action(action["action"])
        explanation = explain_action_with_functions(action, candidates)

        step_result = {
            "action": action["action"],
            "candidates": candidates,
            "decision": explanation,
        }
        full_plan.append(step_result)

        print(f"→ Best Function: {explanation.get('best_function')}")
        print(f"→ Reason: {explanation.get('reason')}\n")

    return full_plan


# === Example Usage ===
if __name__ == "__main__":
    user_query = "add red glow on this clip"
    plan = process_query(user_query)

    print("\n=== FINAL PLAN ===")
    print(json.dumps(plan, indent=2))
