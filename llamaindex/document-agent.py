import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SentenceSplitter
splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)

# Set LLM and embedding model
Settings.llm = Gemini(
    model="models/gemini-2.5-flash",  # Use Gemini model
    api_key=""  # Replace with your Gemini API key
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

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

# Load documents
reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()

# Split documents into nodes
nodes = splitter.get_nodes_from_documents(documents)
logger.info(f"Created {len(nodes)} nodes from documents")

# Build index
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact")

# Debug token count
def log_token_count(query, nodes):
    context = "\n".join([node.text for node in nodes])
    full_input = f"{query}\n{context}"
    # Use HuggingFace tokenizer for token counting
    tokenizer = Settings.embed_model._model.tokenize
    token_count = len(tokenizer(full_input))
    logger.info(f"Total tokens sent to LLM: {token_count}")
    return token_count

# Query
query = "what we can use to create returna and insert component action?"
retrieved_nodes = index.as_retriever(similarity_top_k=2).retrieve(query)
context = "\n".join([node.text for node in retrieved_nodes])
logger.info(f"Query: {query}\nContext sent to LLM:\n{context}")
log_token_count(query, retrieved_nodes)
response = query_engine.query(query)
logger.info(f"Response: {response}")
print(response)