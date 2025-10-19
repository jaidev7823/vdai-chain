import sqlite3
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

SQLITE_DB = "premiere_docs.db"
FAISS_INDEX = "faiss_index"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

def load_query_engine():
    """Load the FAISS index and create query engine"""
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434"
    )
    
    llm = Ollama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        request_timeout=120.0
    )
    
    # Load from disk
    storage_context = StorageContext.from_defaults(persist_dir=FAISS_INDEX)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="compact"
    )
    
    return query_engine

def query_sqlite(query_text):
    """Query SQLite for structured data"""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Example: Find methods by name
    c.execute('''SELECT class_name, item_name, text_content 
                 FROM documents 
                 WHERE item_name LIKE ? AND content_type = 'method'
                 LIMIT 5''', (f'%{query_text}%',))
    
    results = c.fetchall()
    conn.close()
    
    return results

def main():
    print("Loading query engine...")
    query_engine = load_query_engine()
    
    print("\nPremiere Pro Documentation Assistant")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        user_query = input("\nYour question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_query:
            continue
        
        print("\nSearching documentation...")
        
        # Query with LlamaIndex
        response = query_engine.query(user_query)
        print(f"\nAnswer:\n{response}")
        
        # Show source metadata
        if hasattr(response, 'source_nodes'):
            print("\n\nSources:")
            for i, node in enumerate(response.source_nodes, 1):
                metadata = node.node.metadata
                print(f"{i}. {metadata.get('title', 'Unknown')} "
                      f"({metadata.get('section_type', 'Unknown')})")
                print(f"   Relevance: {node.score:.3f}")

if __name__ == "__main__":
    main()