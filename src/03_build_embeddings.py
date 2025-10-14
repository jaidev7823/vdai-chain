# 03_build_embeddings.py
import json
import faiss
from pathlib import Path
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# === Paths ===
DATA_PATH = Path("data/processed/ppro_grouped.json")
FAISS_PATH = Path("embeddings/faiss_index.bin")
STORAGE_PATH = Path("embeddings/llama_storage")

# === Load processed data ===
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

docs = []
for section, objects in data.items():
    for obj, fields in objects.items():
        text = "\n".join(f"{k}: {v}" for k, v in fields.items() if v)
        docs.append(Document(
            text=f"[{section}] {obj}\n{text}",
            metadata={"section": section, "object": obj}
        ))

# === Initialize embedding model (Ollama) ===
embed_model = OllamaEmbedding(model_name="all-minilm")

# === Determine embedding dimension ===
sample_vector = embed_model.get_text_embedding("test")
dimension = len(sample_vector)

# === Create FAISS index ===
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# === Build and persist ===
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)

# Save FAISS and llama storage
faiss.write_index(faiss_index, str(FAISS_PATH))
storage_context.persist(persist_dir=str(STORAGE_PATH))

print("✅ Embeddings built and saved to:")
print(f"   - FAISS index: {FAISS_PATH}")
print(f"   - Llama storage: {STORAGE_PATH}")
