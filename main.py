from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import json
import os

# Load your grouped JSON data
with open("ppro_grouped_with_details.json", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for section, objects in data.items():
    for obj, fields in objects.items():
        text = "\n".join(f"{k}: {v}" for k, v in fields.items() if v)
        docs.append(
            Document(
                text=f"[{section}] {obj}\n{text}",
                metadata={"section": section, "object": obj}
            )
        )

# Initialize Ollama (Mistral)
llm = Ollama(model="mistral", request_timeout=120)
embed_model = OllamaEmbedding(model_name="mistral")

# Build or load index
index_dir = "./ppro_index"
if not os.path.exists(index_dir):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist(index_dir)
else:
    storage = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage)

# Query engine
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("How to use Encoder.startBatch() in Premiere Pro?")
print(response)
