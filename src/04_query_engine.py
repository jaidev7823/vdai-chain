# query_faiss_index.py
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss

# Reload FAISS
faiss_index = faiss.read_index("faiss_index.bin")
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Reload storage + index
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

index = load_index_from_storage(storage_context, embed_model=embed_model)
query_engine = index.as_query_engine()

# Query
response = query_engine.query("How to use Encoder.startBatch() in Premiere Pro?")
print(response)
