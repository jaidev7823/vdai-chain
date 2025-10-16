# 04_query_engine.py
from llama_index.core import StorageContext, load_index_from_storage, PromptTemplate
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import faiss
from pathlib import Path

# === Paths ===
FAISS_PATH = Path("embeddings/faiss_index.bin")
STORAGE_PATH = Path("embeddings/llama_storage")

# === Load FAISS index and storage ===
faiss_index = faiss.read_index(str(FAISS_PATH))
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(STORAGE_PATH))

# === Initialize models ===
embed_model = OllamaEmbedding(model_name="all-minilm")
llm = Ollama(model="mistral", request_timeout=12000)

# === Load LlamaIndex ===
index = load_index_from_storage(storage_context, embed_model=embed_model)

# === Step 1: Retrieve relevant documents ===
query = "squiz this image to on the ai with proper animation like it is croping whole image to only center part and then scale that part to make width of image full"
retriever = index.as_retriever(similarity_top_k=10)
retrieved_nodes = retriever.retrieve(query)

# === Step 2: Generate a response using the retrieved context ===
context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])

prompt_template = PromptTemplate(
    "You are an expert agent for Adobe Premiere Pro. Your task is to create a precise sequence of API calls to accomplish the user\'s request.\n\n"
    "You must only use the functions provided in the API Documentation section. Do not write any code, variables, or logic.\n\n"
    "Your output must be a numbered list of the function names from the documentation.\n\n"
    "User Request: {query_str}\n\n"
    "API Documentation:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Tool Roadmap:"
)

formatted_prompt = prompt_template.format(query_str=query, context_str=context_str)

response = llm.complete(formatted_prompt)

print("\n--- Query ---")
print(query)
print("\n--- Retrieved Documents ---")
for i, node in enumerate(retrieved_nodes):
    print(f"Doc {i+1}: (Score: {node.get_score()})\n{node.get_content()}\n---")
print("\n--- Response ---")
print(response)
