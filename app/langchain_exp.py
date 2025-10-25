# app/langchain_ollama_rag.py
import logging
import os
import faiss
import numpy as np
from pymongo import MongoClient
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
mongo_uri = "mongodb://127.0.0.1:27017/llama_index"
FAISS_PATH = "./embeddings/faiss.index"
DOCSTORE_DB = "llama_index"
DOCSTORE_COLLECTION = "docstore"

# --- 1. LLM + Embeddings ---
llm = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="EmbeddingGemma:latest", base_url="http://localhost:11434")

# --- 2. Load FAISS ---
try:
    faiss_index = faiss.read_index(FAISS_PATH)
    logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    faiss_index = None

# --- 3. Custom Retriever ---
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

client = MongoClient(mongo_uri)
retriever = LlamaIndexSplitRetriever(faiss_index, embeddings, client, k=2)

# --- 4. RAG Prompt ---
template = """Use the following context to answer the question.
If you don't know the answer, just say you don't know.
----------------
CONTEXT:
{context}

QUESTION:
{question}"""

prompt = ChatPromptTemplate.from_template(template)

# --- 5. Build chain ---
def rag_query(question: str):
    docs = retriever.get_relevant_docs(question)
    context_text = "\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.format(context=context_text, question=question)
    return llm.invoke(final_prompt)

# --- 6. Test ---
query = "How to set backend preference?"
response = rag_query(query)
logger.info(f"Question: {query}\nAnswer: {response}")
print(response)
