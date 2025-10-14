# build_faiss_index.py
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss, json

# Load your data
with open("ppro_grouped_with_details.json", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for section, objects in data.items():
    for obj, fields in objects.items():
        text = "\n".join(f"{k}: {v}" for k, v in fields.items() if v)
        docs.append(Document(text=f"[{section}] {obj}\n{text}", metadata={"section": section, "object": obj}))

# Embedding model
embed_model = OpenAIEmbedding(model="all-minilm")

# Get embedding dimension (3072 for this model)
d = embed_model.dimensions

# Create FAISS index
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Attach to LlamaIndex
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build and persist
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
faiss.write_index(faiss_index, "faiss_index.bin")
storage_context.persist("./storage")
print("FAISS index created and saved.")
