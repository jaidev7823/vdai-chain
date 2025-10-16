ppro_ai_assistant/
│
├── data/                         # Raw & processed data
│   ├── raw/                       # Raw scraped JSON/HTML
│   │   └── ppro_docs_raw.json
│   └── processed/                 # Structured/normalized JSON
│       └── ppro_grouped.json
│
├── embeddings/                    # Embedding & vector storage
│   ├── faiss_index.bin            # FAISS binary index
│   └── llama_storage/             # LlamaIndex storage context
│
├── src/                           # All code
│   ├── 01_scrape_docs.py          # Scrape docs from site / JSON
│   ├── 02_parse_structure.py      # Convert raw docs → structured JSON
│   ├── 03_build_embeddings.py     # Convert structured JSON → embeddings + FAISS
│   ├── 04_query_engine.py         # Query engine / test queries
│   └── 05_pipeline.py             # Optional: full pipeline orchestration
│
├── tests/                         # Test scripts for each module
│   └── test_query.py
│
├── config/                        # Configs
│   ├── embeddings_config.json     # e.g., all-mini or other models
│   └── faiss_config.json          # FAISS parameters
│
├── logs/                          # Logs from scraping or queries
│   └── scrape.log
│
├── requirements.txt
└── README.md


Adobe Docs (HTML/JSON)
          │
          ▼
01_scrape_docs.py
          │
          ▼
data/raw/ppro_docs_raw.json
          │
          ▼
02_parse_structure.py
          │
          ▼
data/processed/ppro_grouped.json
          │
          ▼
03_build_embeddings.py
          │
          ├─ embeddings/faiss_index.bin
          └─ embeddings/llama_storage/
          │
          ▼
04_query_engine.py
          │
          ▼
05_pipeline.py
          │
          ▼
API execution plan / instructions


pipeline

User Query
     │
     ▼
Tool Finder (semantic search in FAISS)
     │
     ▼
Candidate Tools (top-k)
     │
     ▼
Context Resolver (LLM)
     │
     ▼
Structured Instructions / Function Call
     │
     ▼
Execution (optional)
