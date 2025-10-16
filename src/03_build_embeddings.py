import json
import sqlite3
import faiss
import numpy as np  # Added import
from pathlib import Path
from llama_index.embeddings.ollama import OllamaEmbedding

# === Paths ===
DATA_PATH = Path("data/processed/ppro_grouped.json")
DB_PATH = Path("embeddings/metadata.db")
FAISS_DIR = Path("embeddings/faiss_indexes")
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# === Init database ===
def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT,
            object_name TEXT,
            member_name TEXT,
            member_type TEXT,
            full_signature TEXT,
            return_type TEXT,
            parameters TEXT,
            faiss_id_description INTEGER,
            faiss_id_details INTEGER,
            faiss_id_example INTEGER
        )
    """)
    conn.commit()

# === Load JSON ===
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# === Initialize embedding model ===
embed_model = OllamaEmbedding(model_name="all-minilm")
dimension = len(embed_model.get_text_embedding("test"))

# === Create FAISS indexes ===
faiss_indexes = {
    "description": faiss.IndexFlatL2(dimension),
    "details": faiss.IndexFlatL2(dimension),
    "example": faiss.IndexFlatL2(dimension),
}

# === Helper: add embedding safely ===
def add_to_faiss(index_name, text):
    if not text or not text.strip():
        return None
    vector = embed_model.get_text_embedding(text)
    vector = np.array(vector, dtype='float32')  # Convert list to numpy array
    idx = faiss_indexes[index_name].ntotal
    faiss_indexes[index_name].add(vector.reshape(1, -1))
    return idx

# === Connect to DB ===
conn = sqlite3.connect(DB_PATH)
init_db(conn)
cur = conn.cursor()

# === Process JSON ===
row_count = 0
for section, objects in data.items():
    for obj, fields in objects.items():
        member_name = obj.split(".")[-1]
        object_name = obj.split(".")[0]
        member_type = "method" if "()" in obj else "property"
        
        desc = fields.get("description", "")
        details = fields.get("details", "")
        example = fields.get("example", "")
        params = fields.get("parameters", "")
        returns = fields.get("returns", "")
        signature = (fields.get("signature") or details or "").replace("<p>", "").replace("</p>", "").strip()
        
        faiss_id_desc = add_to_faiss("description", desc)
        faiss_id_details = add_to_faiss("details", details)
        faiss_id_example = add_to_faiss("example", example)
        
        cur.execute("""
            INSERT INTO metadata (
                section, object_name, member_name, member_type,
                full_signature, return_type, parameters,
                faiss_id_description, faiss_id_details, faiss_id_example
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            section, object_name, member_name, member_type,
            signature, returns, params,
            faiss_id_desc, faiss_id_details, faiss_id_example
        ))
        
        row_count += 1
        if row_count % 200 == 0:
            conn.commit()
            print(f"Processed {row_count} entries...")

conn.commit()
conn.close()

# === Save FAISS indexes ===
for name, index in faiss_indexes.items():
    path = FAISS_DIR / f"{name}.index"
    faiss.write_index(index, str(path))
    print(f"Saved {name}.index → {path}")

print(f"\n✅ Indexed {row_count} metadata entries")
print(f"SQLite DB → {DB_PATH}")