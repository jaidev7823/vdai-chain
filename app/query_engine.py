import logging
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import os
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Set LLM and embedding model
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=api_key)
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma", base_url="http://localhost:11434")

# Initialize Faiss vector store
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Initialize MongoDB stores
mongo_uri = "mongodb://127.0.0.1:27017/llama_index"
docstore = MongoDocumentStore.from_uri(uri=mongo_uri, db_name="llama_index")
index_store = MongoIndexStore.from_uri(uri=mongo_uri, db_name="llama_index")

# Create storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore, index_store=index_store, vector_store=vector_store
)

# Load existing index
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact")

# Debug token count
def log_token_count(query, nodes):
    context = "\n".join([node.text for node in nodes])
    full_input = f"{query}\n{context}"
    tokenizer = Settings.embed_model._model.tokenize
    token_count = len(tokenizer(full_input))
    logger.info(f"Total tokens sent to LLM: {token_count}")
    return token_count

# Query
query = "UXP functions for creating and inserting components"
retrieved_nodes = index.as_retriever(similarity_top_k=2).retrieve(query)
context = "\n".join([node.text for node in retrieved_nodes])
logger.info(f"Query: {query}\nContext sent to LLM:\n{context}")
log_token_count(query, retrieved_nodes)
response = query_engine.query(query)
logger.info(f"Response: {response}")
print(response)