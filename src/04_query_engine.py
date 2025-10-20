#!/usr/bin/env python3
"""
plan_and_pick_api.py  ––  1 file, 2 jobs:
  1. decompose natural-lang request  →  step-by-step plan
  2. for each step pick the single closest Premiere-Pro API
"""

import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# -----------------------------------------------------------
# CONFIG – keep in sync with your existing build scripts
# -----------------------------------------------------------
EMBEDDINGS_DIR = Path("embeddings")
FAISS_DIR      = EMBEDDINGS_DIR / "faiss_indexes"
SQLITE_DB      = EMBEDDINGS_DIR / "premiere_docs.db"
EMBED_MODEL    = "all-minilm"
LLM_MODEL      = "mistral"

PROMPT_FILE    = Path("prompt/prompt.txt")   # your few-shot prompt lives here

# -----------------------------------------------------------
# 1. LLM-based decomposer (identical logic to your first file)
# -----------------------------------------------------------
def decompose_query(query: str) -> list[dict]:
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = f"{system_prompt}\n\nUser request: {query}"

    llm = Ollama(model=LLM_MODEL, request_timeout=12000)
    raw = llm.complete(prompt).text.strip()

    cleaned = (
        raw.replace("```json", "")
           .replace("```", "")
           .replace("Here is a step-by-step guide for the requested action in Premiere Pro:", "")
           .strip()
    )
    try:
        return json.loads(cleaned)          # expect List[{action,description}]
    except json.JSONDecodeError:
        # fallback: each non-empty line becomes an action
        return [{"action": line.strip(), "description": ""}
                for line in cleaned.splitlines() if line.strip()]

# -----------------------------------------------------------
# 2. Documentation searcher (stripped-down version)
# -----------------------------------------------------------
class DocSearcher:
    def __init__(self):
        self.emb = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434")
        self.conn = sqlite3.connect(SQLITE_DB)
        self.index = faiss.read_index(str(FAISS_DIR / "main.index"))

    # ---------- only public method we need ----------
    def nearest_api(self, text: str) -> list[dict]: # Change return type to list
        """Return top-K closest API records for arbitrary text."""
        K = 5 # Set K to 5
        vec = np.array(self.emb.get_text_embedding(text), dtype="float32").reshape(1, -1)
        dists, ids = self.index.search(vec, k=K)

        results = []
        for i in range(K):
            faiss_id = int(ids[0, i])
            distance = float(dists[0, i])

            if faiss_id == -1: # Skip if no more results (e.g., small index)
                continue

            row = self.conn.execute(
                """SELECT class_name, item_name, member_type, full_signature,
                          description, parameters, return_type, details, example_code
                   FROM documents WHERE faiss_id_main = ?""",
                (faiss_id,)
            ).fetchone()

            if row:
                results.append({
                    "class_name": row[0],
                    "item_name": row[1],
                    "member_type": row[2],
                    "full_signature": row[3],
                    "description": row[4] or "",
                    "parameters": row[5] or "",
                    "return_type": row[6] or "",
                    "details": row[7] or "",
                    "example_code": row[8] or "",
                    "similarity": 1.0 / (1.0 + distance) # Inverse distance as score
                })
        return results
# -----------------------------------------------------------
# NEW: LLM-based re-ranker
# -----------------------------------------------------------
def re_rank_apis(action: str, candidates: list[dict]) -> dict:
    """Uses LLM to select the single best API from the top-K candidates."""
    # 1. Format the candidates for the LLM
    candidate_list = "\n".join([
        f"--- Candidate {i+1} ---\n"
        f"API: {c['full_signature']}\n"
        f"Details: {c['details']}\n"
        f"Description: {c['description']}\n"
        for i, c in enumerate(candidates)
    ])

    # 2. Construct the re-ranking prompt
    re_rank_prompt = f"""
    You are an expert Adobe Premiere Pro API developer. Your task is to select the single best API from the provided list of candidates that perfectly matches the user's required action.

    REQUIRED ACTION: "{action}"

    CANDIDATE APIs:
    {candidate_list}

    INSTRUCTION: Review the candidates and output ONLY the 'full_signature' of the single best matching API. Do not add any extra text, explanation, or markdown formatting. The output must be the exact string of the chosen full_signature.
    """

    llm = Ollama(model=LLM_MODEL, request_timeout=12000)
    best_signature = llm.complete(re_rank_prompt).text.strip()

    # 3. Find and return the chosen candidate object
    for c in candidates:
        if c['full_signature'].strip() == best_signature.strip():
            # Add the similarity score from the original Faiss search to the final best_api
            return {**c, "similarity": c.get("similarity", 0.0)}

    # Fallback to the highest-scoring candidate if LLM's output is unusable
    return candidates[0] if candidates else None
# -----------------------------------------------------------
# 3. End-to-end pipeline
# -----------------------------------------------------------
def plan_and_pick(query: str) -> list[dict]:
    plan = decompose_query(query)
    searcher = DocSearcher()
    try:
        for step in plan:
            # Step 1: Get top 5 candidates via semantic search
            candidates = searcher.nearest_api(step["action"])

            if candidates:
                # Step 2: Use LLM to re-rank and pick the best one
                best_api = re_rank_apis(step["action"], candidates)
                step["best_api"] = best_api
            else:
                step["best_api"] = None
    finally:
        searcher.close()
    return plan

# -----------------------------------------------------------
# 4. CLI demo
# -----------------------------------------------------------
if __name__ == "__main__":
    user = (
        "i want my selected image to get crop vertically only 20 px should be visible "
        "and then scale that image so user focus becomes clear"
    )
    out = plan_and_pick(user)
    print(json.dumps(out, indent=2, default=str))