import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SentenceSplitter
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

# Set embedding model
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

# Load and process documents
reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()
nodes = splitter.get_nodes_from_documents(documents)
logger.info(f"Created {len(nodes)} nodes from documents")

# Build and persist index
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)
logger.info("Index created and stored successfully")