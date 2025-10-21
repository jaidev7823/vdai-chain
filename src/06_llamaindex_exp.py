#!/usr/bin/env python3
"""
full_pipeline_llamaindex.py –– fully LlamaIndex-based
  - Ingest JSON docs into LlamaIndex Document objects
  - Create VectorStoreIndex
  - Persist to disk
  - Query with semantic search + optional LLM re-ranking
"""

import json
from pathlib import Path
from llama_index import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# --------------------------------------------
# CONFIG
# --------------------------------------------
DOCS_DIR       = Path("docs_json")
INDEX_DIR      = Path("index_storage")
EMBED_MODEL    = "embeddinggemma"
LLM_MODEL      = "mistral"

# --------------------------------------------
# HELPER: chunk JSON into Document objects
# --------------------------------------------
def create_documents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    class_name = data.get("title", json_file.stem)
    # Overview
    overview_text = f"Class: {class_name}\nDescription: {data.get('description','')}"
    docs.append(Document(text=overview_text, metadata={"class_name": class_name, "section": "overview"}))

    # Methods / Commands
    for section in data.get("sections", []):
        for name, content in section.items():
            if isinstance(content, dict) and "commands" in content:
                for cmd in content["commands"]:
                    cmd_name = cmd.get("command", {}).get("name", "unknown")
                    desc = cmd.get("command", {}).get("description", "")
                    main_text = f"{class_name}.{cmd_name}\n{desc}"
                    docs.append(Document(text=main_text, metadata={"class_name": class_name, "section": name, "item": cmd_name}))
            elif name == "Enumerations" and isinstance(content, dict):
                for enum_name, enum_values in content.items():
                    if enum_name != "content":
                        text = f"{class_name}.{enum_name}\nValues: {enum_values}"
                        docs.append(Document(text=text, metadata={"class_name": class_name, "section": "enum", "item": enum_name}))
    return docs

# --------------------------------------------
# BUILD INDEX
# --------------------------------------------
def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Embeddings
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434")
    service_ctx = ServiceContext.from_defaults(embed_model=embed_model)

    all_docs = []
    for f in DOCS_DIR.glob("*.json"):
        all_docs.extend(create_documents(f))

    # Vector index
    index = VectorStoreIndex.from_documents(all_docs, service_context=service_ctx)

    # Persist
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index.storage_context = storage_context
    index.storage_context.persist()
    print(f"✅ Indexed {len(all_docs)} documents into {INDEX_DIR}")

# --------------------------------------------
# QUERY ENGINE
# --------------------------------------------
def query_index(query: str, top_k: int = 5):
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url="http://localhost:11434")
    service_ctx = ServiceContext.from_defaults(embed_model=embed_model)
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = VectorStoreIndex.load_from_storage(storage_context, service_context=service_ctx)
    # LlamaIndex query engine
    response_nodes = index.as_query_engine(similarity_top_k=top_k).retrieve(query)
    return [{"text": n.node.get_text(), "metadata": n.node.metadata} for n in response_nodes]

# --------------------------------------------
# LLM RE-RANKER (optional)
# --------------------------------------------
def re_rank(query: str, candidates: list[dict]):
    candidate_text = "\n".join([f"{c['text']}" for c in candidates])
    prompt = f"""
You are an expert Premiere Pro API developer. User wants: "{query}"

Candidate APIs:
{candidate_text}

Choose the single best API by returning ONLY the full_signature string.
"""
    llm = Ollama(model=LLM_MODEL, request_timeout=12000)
    best = llm.complete(prompt).text.strip()
    for c in candidates:
        if c['text'].startswith(best):
            return c
    return candidates[0] if candidates else None

# --------------------------------------------
# CLI
# --------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Build index from JSON docs")
    parser.add_argument("--query", type=str, help="Query API docs")
    parser.add_argument("--rerank", action="store_true", help="Use LLM to pick best API")
    args = parser.parse_args()

    if args.index:
        build_index()

    if args.query:
        candidates = query_index(args.query, top_k=5)
        if args.rerank:
            best = re_rank(args.query, candidates)
            print(json.dumps(best, indent=2))
        else:
            print(json.dumps(candidates, indent=2))
