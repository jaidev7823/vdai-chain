import logging
import os
from dotenv import load_dotenv
import faiss

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# LLM and embedding model
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=api_key)
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma", base_url="http://localhost:11434")

# MongoDB stores
mongo_uri = "mongodb://127.0.0.1:27017/llama_index"
docstore = MongoDocumentStore.from_uri(uri=mongo_uri, db_name="llama_index")
index_store = MongoIndexStore.from_uri(uri=mongo_uri, db_name="llama_index")

# Load persisted Faiss index
embedding_dim = 768
faiss_index = faiss.read_index("./embeddings/faiss.index")
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
)

# Load index fully (nodes + Faiss vectors)
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=10, response_mode="compact")

# Query
query = "Where is the AnAnywhere object available" 
retrieved_nodes = index.as_retriever(similarity_top_k=10).retrieve(query)
context = "\n".join([node.text for node in retrieved_nodes])
logger.info(f"Query: {query}\nContext:\n{context}")

response = query_engine.query(query)
logger.info(f"Response: {response}")
print(response)
