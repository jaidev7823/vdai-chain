import logging
import os
from dotenv import load_dotenv
import faiss
from pymongo import MongoClient

from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ---- Embedding model ----
embed_model = OllamaEmbedding(model_name="embeddinggemma", base_url="http://localhost:11434")
Settings.embed_model = embed_model

# ---- Detect embedding dimension dynamically ----
sample_vec = embed_model.get_text_embedding("test")
embedding_dim = len(sample_vec)
logger.info(f"Detected embedding dimension: {embedding_dim}")

# ---- MongoDB setup ----
mongo_uri = "mongodb://127.0.0.1:27017"
db_name = "llama_index"

MongoClient(mongo_uri).drop_database(db_name)
logger.info("Old MongoDB database dropped")

docstore = MongoDocumentStore.from_uri(uri=mongo_uri, db_name=db_name)
index_store = MongoIndexStore.from_uri(uri=mongo_uri, db_name=db_name)

# ---- FAISS setup ----
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Ensure embeddings folder exists
os.makedirs("./embeddings", exist_ok=True)
faiss_file_path = "./embeddings/faiss.index"

# ---- Storage Context ----
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
)

# ---- Read and index documents ----
documents = SimpleDirectoryReader(input_dir="./docs_txt", recursive=True).load_data()
logger.info(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

# ---- Persist everything ----
storage_context.persist()
faiss.write_index(vector_store._faiss_index, faiss_file_path)

logger.info(f"FAISS index saved at {faiss_file_path}")
logger.info(f"Total vectors in FAISS: {vector_store._faiss_index.ntotal}")

logger.info("Index creation and storage complete")
