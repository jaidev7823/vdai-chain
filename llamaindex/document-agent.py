from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
import os

# Initialize SQLite database path
sqlite_db_path = "storage.db"

# Initialize Faiss vector store
embedding_dim = 768  # Adjust based on your embedding model (e.g., BAAI/bge-base-en-v1.5)
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Initialize Redis stores
doc_store = RedisDocumentStore.from_default()
index_store = RedisIndexStore.from_default()
kv_store = RedisKVStore.from_default()
chat_store = RedisChatStore.from_default()

# Create storage context
storage_context = StorageContext.from_defaults(
    docstore=doc_store,
    index_store=index_store,
    kv_store=kv_store,
    vector_store=vector_store,
    chat_store=chat_store
)

# Load documents from directory
reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()

# Parse documents into nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

# Use HuggingFace embeddings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Build index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

# Save index to storage
index.storage_context.persist(persist_dir="./storage")

# Reload existing index (example)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
    storage_context=storage_context
)