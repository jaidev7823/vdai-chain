# 04_query_engine.py
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import faiss
from pathlib import Path

# === Paths ===
FAISS_PATH = Path("embeddings/faiss_index.bin")
STORAGE_PATH = Path("embeddings/llama_storage")

# === Load FAISS index and storage ===
faiss_index = faiss.read_index(str(FAISS_PATH))
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(STORAGE_PATH))

# === Initialize models ===
embed_model = OllamaEmbedding(model_name="all-minilm")
llm = Ollama(model="mistral", request_timeout=12000)

# === Load LlamaIndex ===
index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

# === Query ===
query = "How you can you add red like effect on video?"
response = query_engine.query(query)

print("\n--- Query ---")
print(query)
print("\n--- Response ---")
print(response)
