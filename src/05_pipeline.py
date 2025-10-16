# src/05_pipeline.py
import json
from sentence_transformers import SentenceTransformer
import faiss
from llama_index.llms.ollama import Ollama
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
FAISS_INDEX_PATH = "embeddings/faiss_index.bin"
DOCS_PATH = "data/processed/ppro_grouped.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"  # or any Ollama model
TOP_K = 5
SIM_THRESHOLD = 0.55

# ----------------------------
# LOADERS
# ----------------------------
def load_faiss_index(path=FAISS_INDEX_PATH):
    index = faiss.read_index(path)
    return index

def load_docs(path=DOCS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for section, objs in raw.items():
        for name, fields in objs.items():
            docs.append({
                "name": name,
                "description": fields.get("details", ""),
                "example": fields.get("example", ""),
                "section": section
            })
    return docs


# ----------------------------
# SEMANTIC TOOL FINDER
# ----------------------------
def find_relevant_tools(query, index, docs, model, top_k=TOP_K, threshold=SIM_THRESHOLD):
    q_emb = np.array([model.encode(query, convert_to_numpy=True)]).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if dist >= threshold:
            results.append({
                "similarity": float(dist),
                "tool": docs[idx].get("name", "unknown"),
                "description": docs[idx].get("description", ""),
                "example": docs[idx].get("example", "")
            })
    return results

# ----------------------------
# LLM CLARIFIER
# ----------------------------
def clarify_query(query, tool_candidates, llm):
    tool_list = "\n".join(
        [f"- {t['tool']}: {t['description']}" for t in tool_candidates]
    )
    prompt = f"""
    User query: {query}
    
    Here are the top relevant tool descriptions:
    {tool_list}
    
    You are the PLANNER LLM.
    Output a JSON plan describing which tools should be called, in what order, and which parameters should be filled.
    Do NOT give human instructions. Do NOT describe UI steps.
    
    JSON keys:
    - goal
    - tools: [{{
        name: str (tool or API name),
        purpose: str (why used),
        parameters: dict (parameter name: value or source)
      }}]
    """

    resp = llm.complete(prompt)
    return resp.text.strip()

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_pipeline(query):
    print(f"\n--- Running pipeline for query ---\n{query}\n")

    # Load models & index
    model = SentenceTransformer(EMBED_MODEL)
    llm = Ollama(model=LLM_MODEL, request_timeout=1200)
    index = load_faiss_index()
    docs = load_docs()

    # Step 1: Tool finder
    print("Searching for relevant tools...")
    tools = find_relevant_tools(query, index, docs, model)
    if not tools:
        print("No relevant tools found.")
        return

    # Step 2: LLM clarification
    print("Clarifying context and action...")
    answer = clarify_query(query, tools, llm)
    print("\n--- Clarified Instruction ---")
    print(answer)

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    run_pipeline(user_query)
