import logging
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.gemini import Gemini
import os
from dotenv import load_dotenv
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# LLM and embedding model
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=api_key)
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma", base_url="http://localhost:11434")

# MongoDB stores
mongo_uri = "mongodb://127.0.0.1:27017/llama_index"
docstore = MongoDocumentStore.from_uri(uri=mongo_uri, db_name="llama_index")
index_store = MongoIndexStore.from_uri(uri=mongo_uri, db_name="llama_index")

# Faiss vector store
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
)

# Load existing index from storage
index = VectorStoreIndex.load_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact")

# Query example
query = "UXP functions for creating and inserting components"
retrieved_nodes = index.as_retriever(similarity_top_k=2).retrieve(query)
context = "\n".join([node.text for node in retrieved_nodes])
logger.info(f"Query: {query}\nContext:\n{context}")

response = query_engine.query(query)
logger.info(f"Response: {response}")
print(response)
