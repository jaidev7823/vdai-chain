# docs igection for that we will need SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./docs_txt")
documents = reader.load_data()

## build a new index
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

# construct vector store and customize storage context
vector_store = DeepLakeVectorStore(dataset_path="/embeddings")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Load documents and build index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)


## reload an existing one
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)