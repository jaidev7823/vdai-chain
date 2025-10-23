from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
import os
from pymongo import MongoClient

# Initialize Faiss vector store
embedding_dim = 768  # Adjust based on your embedding model
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Initialize MongoDB stores
client = MongoClient("mongodb://127.0.0.1:27017")
db_name = "llama_index"

docstore = MongoDocumentStore(
    client=client,
    db_name=db_name,
    collection_name="documents"
)

index_store = MongoIndexStore(
    client=client,
    db_name=db_name,
    collection_name="indexes"
)

# Create storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
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

# Persist storage context if needed (for Faiss + Mongo)
index.storage_context.persist(persist_dir="./storage")

# Reload existing index (example)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
    storage_context=storage_context
)
