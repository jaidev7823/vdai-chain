from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage,Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
from pymongo import MongoClient
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Initialize Faiss vector store
embedding_dim = 768  # For BAAI/bge-base-en-v1.5
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

# Load documents
reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()

# Use HuggingFace embeddings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Build index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What does createAppendComponentAction do?")
print(response)