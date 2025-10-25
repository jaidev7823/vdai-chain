# app/langchain_ollama_rag_test.py
import logging
import os
import faiss
import numpy as np
from pymongo import MongoClient
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
mongo_uri = "mongodb://127.0.0.1:27017/llama_index"
FAISS_PATH = "./embeddings/faiss.index"
DOCSTORE_DB = "llama_index"
DOCSTORE_COLLECTION = "docstore"

# --- 1. Embeddings ---
embeddings = OllamaEmbeddings(model="EmbeddingGemma:latest", base_url="http://localhost:11434")

# --- 2. Load FAISS ---
try:
    faiss_index = faiss.read_index(FAISS_PATH)
    logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    faiss_index = None

# --- 3. Retriever ---
class LlamaIndexSplitRetriever:
    def __init__(self, faiss_index, embeddings, mongo_client, k=2):
        self.faiss_index = faiss_index
        self.embeddings = embeddings
        self.mongo_client = mongo_client
        self.k = k

    def get_relevant_docs(self, query: str):
        query_vector = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
        D, I = self.faiss_index.search(query_vector, self.k)
        db = self.mongo_client[DOCSTORE_DB]
        collection = db[DOCSTORE_COLLECTION]
        docs = []
        for idx in I[0]:
            doc_data = collection.find_one({"_id": int(idx.item())})  # ensures pure int
            if doc_data:
                docs.append(Document(
                    page_content=doc_data.get("text", ""),
                    metadata=doc_data.get("metadata", {})
                ))
        return docs

# --- 4. Test ---
client = MongoClient(mongo_uri)
retriever = LlamaIndexSplitRetriever(faiss_index, embeddings, client, k=2)

query = "How to set backend preference?"
docs = retriever.get_relevant_docs(query)

logger.info(f"Query: {query}")
for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
