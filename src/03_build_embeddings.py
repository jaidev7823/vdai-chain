import os
import json
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ============================================================================
# CONFIGURATION
# ============================================================================
DOCS_DIR = Path("docs_json")                       # JSON files from scraper
PROCESSED_DIR = Path("data/processed")             # Processed chunks
EMBEDDINGS_DIR = Path("embeddings")                # All embedding outputs
FAISS_DIR = EMBEDDINGS_DIR / "faiss_indexes"      # FAISS indexes
SQLITE_DB = EMBEDDINGS_DIR / "premiere_docs.db"    # SQLite metadata

# Create directories
for dir_path in [PROCESSED_DIR, EMBEDDINGS_DIR, FAISS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
EMBED_MODEL = "all-minilm"  # Fast and good quality
LLM_MODEL = "llama3.2"      # or "mistral", "codellama"

# ============================================================================
# DATABASE SETUP
# ============================================================================
def create_sqlite_db():
    """Create SQLite database with optimized schema"""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Main documents table with FAISS mapping
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doc_id TEXT UNIQUE,
                  class_name TEXT,
                  section_type TEXT,
                  item_name TEXT,
                  member_type TEXT,
                  full_signature TEXT,
                  description TEXT,
                  return_type TEXT,
                  parameters TEXT,
                  details TEXT,
                  example_code TEXT,
                  faiss_id_main INTEGER,
                  faiss_id_description INTEGER,
                  faiss_id_details INTEGER,
                  faiss_id_example INTEGER,
                  json_metadata TEXT)''')
    
    # Parameters table for granular search
    c.execute('''CREATE TABLE IF NOT EXISTS parameters
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  doc_id TEXT,
                  class_name TEXT,
                  method_name TEXT,
                  param_name TEXT,
                  param_type TEXT,
                  param_description TEXT,
                  FOREIGN KEY(doc_id) REFERENCES documents(doc_id))''')
    
    # Create indexes for fast lookups
    c.execute('CREATE INDEX IF NOT EXISTS idx_class ON documents(class_name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_section ON documents(section_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_member ON documents(item_name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_type ON documents(member_type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_method_params ON parameters(method_name)')
    
    conn.commit()
    print(f"✅ SQLite database created: {SQLITE_DB}")
    return conn

# ============================================================================
# FAISS SETUP - Multiple indexes for different content types
# ============================================================================
def create_faiss_indexes(dimension):
    """Create separate FAISS indexes for different content types"""
    faiss_indexes = {
        "main": faiss.IndexFlatL2(dimension),           # Full content
        "description": faiss.IndexFlatL2(dimension),    # Short descriptions
        "details": faiss.IndexFlatL2(dimension),        # Detailed info
        "example": faiss.IndexFlatL2(dimension),        # Code examples
    }
    print(f"✅ Created FAISS indexes with dimension {dimension}")
    return faiss_indexes

def add_to_faiss(faiss_indexes, index_name, text, embed_model):
    """Safely add text embedding to FAISS index"""
    if not text or not text.strip():
        return None
    
    try:
        vector = embed_model.get_text_embedding(text)
        vector = np.array(vector, dtype='float32').reshape(1, -1)
        idx = faiss_indexes[index_name].ntotal
        faiss_indexes[index_name].add(vector)
        return idx
    except Exception as e:
        print(f"Warning: Failed to embed text for {index_name}: {e}")
        return None

# ============================================================================
# CONTENT FORMATTING
# ============================================================================
def format_table_as_text(table_data):
    """Convert table to readable text"""
    if not isinstance(table_data, dict) or "rows" not in table_data:
        return str(table_data)
    
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    
    if not rows:
        return ""
    
    lines = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append("-" * 50)
    
    for row in rows:
        if isinstance(row, dict):
            lines.append(" | ".join(str(row.get(h, "")) for h in headers))
        else:
            lines.append(" | ".join(str(cell) for cell in row))
    
    return "\n".join(lines)

def extract_code_examples(details):
    """Extract code examples from details"""
    examples = []
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict) and "code" in item:
                examples.append(item["code"])
    return "\n\n".join(examples)

def extract_parameters_info(parameters):
    """Extract parameter information as text"""
    if not parameters or not isinstance(parameters, list):
        return ""
    
    param_lines = []
    for param in parameters:
        if isinstance(param, dict):
            name = param.get("Name", param.get("Parameter", ""))
            ptype = param.get("Type", "")
            desc = param.get("Description", "")
            param_lines.append(f"{name} ({ptype}): {desc}")
    
    return "\n".join(param_lines)

# ============================================================================
# CHUNK CREATION
# ============================================================================
def create_method_chunk(class_name, section_type, command_data):
    """Create comprehensive chunk for method/property"""
    cmd = command_data.get("command", {})
    name = cmd.get("name", "Unknown")
    description = cmd.get("description", "")
    parameters = cmd.get("parameters", [])
    returns = cmd.get("returns", [])
    details = cmd.get("details", [])
    
    # Extract different content types
    params_text = extract_parameters_info(parameters)
    
    returns_text = ""
    if returns:
        if isinstance(returns, list):
            for ret in returns:
                if isinstance(ret, dict):
                    returns_text += f"{ret.get('Type', '')}: {ret.get('Description', '')}\n"
    
    details_text_parts = []
    example_code = ""
    
    if details:
        for detail in details:
            if isinstance(detail, dict):
                if "content" in detail:
                    details_text_parts.append(detail["content"])
                elif "Table" in detail:
                    details_text_parts.append(format_table_as_text(detail["Table"]))
                elif "code" in detail:
                    example_code += detail["code"] + "\n\n"
                elif "list" in detail:
                    details_text_parts.extend([f"- {item}" for item in detail["list"]])
    
    details_text = "\n".join(details_text_parts)
    
    # Build full signature
    member_type = "method" if parameters else "property"
    if member_type == "method":
        param_names = [p.get("Name", p.get("Parameter", "")) for p in parameters if isinstance(p, dict)]
        signature = f"{class_name}.{name}({', '.join(param_names)})"
    else:
        signature = f"{class_name}.{name}"
    
    # Build comprehensive main text
    main_text_parts = [
        f"Class: {class_name}",
        f"Type: {section_type}",
        f"Signature: {signature}",
    ]
    
    if description:
        main_text_parts.append(f"Description: {description}")
    
    if params_text:
        main_text_parts.append(f"\nParameters:\n{params_text}")
    
    if returns_text:
        main_text_parts.append(f"\nReturns:\n{returns_text}")
    
    if details_text:
        main_text_parts.append(f"\nDetails:\n{details_text}")
    
    if example_code:
        main_text_parts.append(f"\nExample:\n{example_code}")
    
    main_text = "\n".join(main_text_parts)
    doc_id = f"{class_name}.{name}".replace(" ", "_")
    
    return {
        "doc_id": doc_id,
        "class_name": class_name,
        "section_type": section_type,
        "item_name": name,
        "member_type": member_type,
        "full_signature": signature,
        "description": description,
        "return_type": returns_text.strip(),
        "parameters": params_text,
        "details": details_text,
        "example_code": example_code.strip(),
        "main_text": main_text,
        "parameters_list": parameters,
        "json_metadata": json.dumps(command_data)
    }

def create_enum_chunk(class_name, enum_name, enum_values):
    """Create chunk for enumeration"""
    values_text = []
    
    if isinstance(enum_values, list):
        for item in enum_values:
            if isinstance(item, dict):
                if "content" in item:
                    values_text.append(item["content"])
                elif "Table" in item:
                    values_text.append(format_table_as_text(item["Table"]))
    
    description = f"Enumeration {enum_name} with possible values"
    details = "\n".join(values_text)
    
    main_text = f"Class: {class_name}\nEnumeration: {enum_name}\n\nValues:\n{details}"
    doc_id = f"{class_name}.{enum_name}".replace(" ", "_")
    
    return {
        "doc_id": doc_id,
        "class_name": class_name,
        "section_type": "Enumeration",
        "item_name": enum_name,
        "member_type": "enum",
        "full_signature": f"{class_name}.{enum_name}",
        "description": description,
        "return_type": "",
        "parameters": "",
        "details": details,
        "example_code": "",
        "main_text": main_text,
        "parameters_list": [],
        "json_metadata": json.dumps({"name": enum_name, "values": enum_values})
    }

def create_overview_chunk(file_data):
    """Create chunk for class overview"""
    title = file_data.get("title", "Unknown")
    description = file_data.get("description", "")
    
    details_parts = []
    
    for section in file_data.get("sections", []):
        for section_name, section_data in section.items():
            if section_name in ["Overview", "System requirements", "Reference material"]:
                details_parts.append(f"{section_name}:")
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    if "content" in item:
                                        details_parts.append(item["content"])
                                    elif "list" in item:
                                        details_parts.extend([f"- {i}" for i in item["list"]])
    
    details = "\n".join(details_parts)
    main_text = f"Class: {title}\nDescription: {description}\n\n{details}"
    doc_id = f"{title}.Overview"
    
    return {
        "doc_id": doc_id,
        "class_name": title,
        "section_type": "Overview",
        "item_name": "Overview",
        "member_type": "overview",
        "full_signature": title,
        "description": description,
        "return_type": "",
        "parameters": "",
        "details": details,
        "example_code": "",
        "main_text": main_text,
        "parameters_list": [],
        "json_metadata": json.dumps({"title": title, "description": description})
    }

# ============================================================================
# MAIN PROCESSING
# ============================================================================
def process_json_files(conn, faiss_indexes, embed_model):
    """Process all JSON files and create embeddings"""
    chunks = []
    c = conn.cursor()
    
    json_files = list(DOCS_DIR.glob("*.json"))
    print(f"\n📄 Found {len(json_files)} JSON files to process")
    
    for idx, json_file in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] Processing {json_file.name}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        class_name = data.get("title", json_file.stem)
        
        # Create overview chunk
        overview_chunk = create_overview_chunk(data)
        chunks.append(overview_chunk)
        
        # Process sections
        for section in data.get("sections", []):
            for section_name, section_data in section.items():
                
                # Handle method/property sections with commands
                if isinstance(section_data, dict) and "commands" in section_data:
                    for command in section_data["commands"]:
                        chunk = create_method_chunk(class_name, section_name, command)
                        chunks.append(chunk)
                
                # Handle enumerations
                elif section_name == "Enumerations" and isinstance(section_data, dict):
                    for enum_name, enum_values in section_data.items():
                        if enum_name != "content":
                            chunk = create_enum_chunk(class_name, enum_name, enum_values)
                            chunks.append(chunk)
    
    print(f"\n✅ Created {len(chunks)} chunks")
    print("\n🔢 Creating embeddings and inserting into database...")
    
    # Process chunks in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        for chunk in batch:
            # Create embeddings for different content types
            faiss_id_main = add_to_faiss(faiss_indexes, "main", chunk["main_text"], embed_model)
            faiss_id_desc = add_to_faiss(faiss_indexes, "description", chunk["description"], embed_model)
            faiss_id_details = add_to_faiss(faiss_indexes, "details", chunk["details"], embed_model)
            faiss_id_example = add_to_faiss(faiss_indexes, "example", chunk["example_code"], embed_model)
            
            # Insert into database
            c.execute('''INSERT OR REPLACE INTO documents 
                       (doc_id, class_name, section_type, item_name, member_type,
                        full_signature, description, return_type, parameters, details, example_code,
                        faiss_id_main, faiss_id_description, faiss_id_details, faiss_id_example, json_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (chunk["doc_id"], chunk["class_name"], chunk["section_type"], 
                      chunk["item_name"], chunk["member_type"], chunk["full_signature"],
                      chunk["description"], chunk["return_type"], chunk["parameters"], 
                      chunk["details"], chunk["example_code"],
                      faiss_id_main, faiss_id_desc, faiss_id_details, faiss_id_example,
                      chunk["json_metadata"]))
            
            # Insert parameters
            for param in chunk["parameters_list"] or []:
                if isinstance(param, dict):
                    c.execute('''INSERT INTO parameters 
                               (doc_id, class_name, method_name, param_name, param_type, param_description)
                               VALUES (?, ?, ?, ?, ?, ?)''',
                             (chunk["doc_id"], chunk["class_name"], chunk["item_name"],
                              param.get("Name", param.get("Parameter", "")),
                              param.get("Type", ""),
                              param.get("Description", "")))
        
        conn.commit()
        print(f"  Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks...")
    
    print(f"✅ Inserted {len(chunks)} entries into SQLite")
    
    return chunks

def save_faiss_indexes(faiss_indexes):
    """Save all FAISS indexes to disk"""
    print("\n💾 Saving FAISS indexes...")
    for name, index in faiss_indexes.items():
        path = FAISS_DIR / f"{name}.index"
        faiss.write_index(index, str(path))
        print(f"  ✓ Saved {name}.index ({index.ntotal} vectors)")
    
    print(f"✅ All FAISS indexes saved to {FAISS_DIR}")

def save_processed_chunks(chunks):
    """Save processed chunks as JSON for backup/debugging"""
    output_path = PROCESSED_DIR / "processed_chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved processed chunks to {output_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("🚀 Adobe Premiere Pro Documentation Indexer")
    print("=" * 70)
    
    # Step 1: Initialize embedding model
    print("\n📦 Initializing embedding model...")
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434"
    )
    
    # Get embedding dimension
    test_vector = embed_model.get_text_embedding("test")
    dimension = len(test_vector)
    print(f"✅ Embedding model loaded (dimension: {dimension})")
    
    # Step 2: Create databases
    print("\n🗄️  Creating SQLite database...")
    conn = create_sqlite_db()
    
    print("\n🔍 Creating FAISS indexes...")
    faiss_indexes = create_faiss_indexes(dimension)
    
    # Step 3: Process JSON files
    print("\n⚙️  Processing documentation files...")
    chunks = process_json_files(conn, faiss_indexes, embed_model)
    
    # Step 4: Save everything
    save_faiss_indexes(faiss_indexes)
    save_processed_chunks(chunks)
    
    conn.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ INDEXING COMPLETE!")
    print("=" * 70)
    print(f"📊 Total chunks indexed: {len(chunks)}")
    print(f"📁 SQLite database: {SQLITE_DB}")
    print(f"📁 FAISS indexes: {FAISS_DIR}")
    print(f"📁 Processed data: {PROCESSED_DIR}")
    print("\n💡 Next steps:")
    print("   1. Run the query script to test search")
    print("   2. Integrate with your application")
    print("=" * 70)

if __name__ == "__main__":
    main()