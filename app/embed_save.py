import logging
import os
from dotenv import load_dotenv
import faiss
from pymongo import MongoClient

from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.ollama import OllamaEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Embedding model
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma", base_url="http://localhost:11434")

# MongoDB setup
mongo_uri = "mongodb://127.0.0.1:27017"
db_name = "llama_index"

# Drop old database if it exists
MongoClient(mongo_uri).drop_database(db_name)
logger.info("Old MongoDB database dropped")

docstore = MongoDocumentStore.from_uri(uri=mongo_uri, db_name=db_name)
index_store = MongoIndexStore.from_uri(uri=mongo_uri, db_name=db_name)

# Faiss setup
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Ensure embeddings folder exists
embeddings_folder = "./embeddings"
os.makedirs(embeddings_folder, exist_ok=True)
faiss_file_path = os.path.join(embeddings_folder, "faiss.index")

# Storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
)

# Read and split documents
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()
nodes = splitter.get_nodes_from_documents(documents)
logger.info(f"Created {len(nodes)} nodes from documents")

# Build VectorStoreIndex from documents
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=Settings.embed_model,
    node_parser=splitter
)

# Persist Faiss to disk
faiss.write_index(faiss_index, faiss_file_path)
logger.info(f"Faiss index persisted at {faiss_file_path}")

logger.info("Index creation and storage complete")
