import logging
import os
import faiss

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM and embedding model (local Ollama)
llm = Ollama(model="phi3:latest")  # local model
embed_model = OllamaEmbedding(model_name="embeddinggemma:latest", base_url="http://localhost:11434")  # embeddings still local

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
index = load_index_from_storage(
    storage_context,
    embed_model=embed_model
)
# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact",llm=llm)

# Query
query = "how to set transition of duration"
retrieved_nodes = index.as_retriever(similarity_top_k=2).retrieve(query)
context = "\n".join([node.text for node in retrieved_nodes])
logger.info(f"Query: {query}\nContext:\n{context}")

response = query_engine.query(query)
logger.info(f"Response: {response}")
print(response)
