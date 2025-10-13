from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from langchain.schema import SystemMessage, HumanMessage

import os

# ---------- Step 1: Load / Build Index ----------
index_path = "index_store"

if not os.path.exists(index_path):
    print("Building index...")
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path)
else:
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=3)

# ---------- Step 2: Define LLM ----------
llm = Ollama(model="mistral")

# ---------- Step 3: Define Test Prompt ----------
user_prompt = "Move the first clip on timeline to 5 seconds and in the gap import a file named intro.mp4"

# ---------- Step 4: Retrieve Relevant Docs ----------
retrieved_context = query_engine.query(user_prompt)
context_text = retrieved_context.response

# ---------- Step 5: Ask Model to Choose Tool ----------
system_prompt = """You are an AI that decides which Adobe Premiere API should be used for a given task.
You are given a task and a list of related documentation snippets.
Respond only in JSON:
{
  "tool": "API_name",
  "reason": "why this API fits the task",
  "parameters": {...}
}
"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"Task: {user_prompt}\n\nDocs:\n{context_text}")
]

response = llm.invoke(messages)
print("\n--- Model Decision ---")
print(response)
