# RetrievalService

RAG pipeline: document parsing (Docling), chunking, Vertex AI embeddings (gemini-embedding-001), and cosine similarity search over pgvector.

## Setup

### Infrastructure

In `sunset.yaml`:

```yaml
infra:
  llm:
    provider: vertexai
```

This enables the Vertex AI API (needed for embeddings). No `filestores` needed — RetrievalService uses pgvector, not Discovery Engine.

### Database

Requires the `pgvector` extension and a `knowledge_chunks` table. Create a migration:

```bash
sunset migrate --create -m "add knowledge chunks"
```

Migration content:

```python
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("source_file", sa.String, nullable=False),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("content_type", sa.String, default="text_chunk"),
        sa.Column("headings", sa.ARRAY(sa.String), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute("""
        CREATE INDEX knowledge_chunks_embedding_idx
        ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    """)

def downgrade():
    op.drop_table("knowledge_chunks")
```

### Extra Dependencies

The child project's `requirements.txt` needs:

```
pgvector
docling
docling-core
```

These are heavy optional dependencies not bundled with sunset by default.

## Usage

```python
from sunset.services.retrieval import RetrievalService

retrieval = RetrievalService(
    dsn=DATABASE_URL,
    project=GCP_PROJECT_ID,
    location="europe-west1",
)

# Connect on startup
await retrieval.connect()

# Ingest a document (PDF, DOCX, PPTX, HTML, Markdown)
chunks_count = await retrieval.ingest_document(
    file_path="/tmp/uploaded.pdf",
    metadata={"school_id": "abc123"},
)

# Fast ingestion — skip OCR and table structure for text-layer PDFs
chunks_count = await retrieval.ingest_document(
    file_path="/tmp/uploaded.pdf",
    metadata={"school_id": "abc123"},
    do_ocr=False,
    do_table_structure=False,
)

# Ingest raw text
chunks_count = await retrieval.ingest(
    text="Some raw content...",
    source_file="notes.txt",
)

# Query for similar chunks
results = await retrieval.query("What is the refund policy?", top_k=5)

# Query with metadata filter (dict — safe, parameterised)
results = await retrieval.query(
    "What is the refund policy?",
    top_k=5,
    where={"school_id": "abc123"},
)

# Filter with operators (source_file and content_type target columns directly,
# all other keys target the metadata JSONB column)
results = await retrieval.query(
    "symptoms",
    top_k=5,
    where={
        "source_file": "patient_records.pdf",
        "doctor_id": {"$in": ["dr_1", "dr_2"]},
        "confidence": {"$gte": 0.8},
        "status": {"$ne": "archived"},
    },
)

# Raw SQL filter (caller responsible for safety)
results = await retrieval.query(
    "symptoms",
    top_k=5,
    where="metadata->>'department' ILIKE '%cardio%'",
)

for chunk in results:
    print(chunk["content"], chunk["score"])

# List all ingested source files
files = await retrieval.list_sources()

# List sources filtered by metadata
files = await retrieval.list_sources(where={"school_id": "abc123"})
# [{"source_file": "report.pdf", "chunks_count": 49, "created_at": datetime(...)}, ...]

# Delete chunks by metadata
deleted = await retrieval.delete(where={"school_id": "abc123"})

# Delete chunks by source file (source_file and content_type target columns directly)
deleted = await retrieval.delete(where={"source_file": "old_doc.pdf"})
deleted = await retrieval.delete(where={"source_file": "report.pdf", "doctor_id": "dr_123"})

# Cleanup on shutdown
await retrieval.close()
```

### Integration with LLM + ChatService

```python
from sunset.services.llm import VertexAIGeminiService, file_search

llm = VertexAIGeminiService(
    project=PROJECT_ID,
    location="global",
    retrieval=retrieval,  # Pass retrieval service
)

# file_search tool uses retrieval.query() under the hood
chat = ChatService(llm=llm, tools=[file_search], ...)
```

## API Reference

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dsn` | `str` | required | PostgreSQL connection string |
| `project` | `str` | required | GCP project ID (for Vertex AI embeddings) |
| `location` | `str` | `"europe-west1"` | GCP region for Vertex AI |

### Key Methods

- `connect()` / `close()` — Manage the asyncpg connection pool (async)
- `embed(text) -> list[float]` — Embed a single text (async)
- `embed_batch(texts) -> list[list[float]]` — Batch embed (async)
- `ingest(text, source_file, metadata?, max_tokens?) -> int` — Chunk and embed raw text (async)
- `ingest_document(file_path, metadata?, describe_images?, max_tokens?, do_ocr=True, do_table_structure=True, num_threads=4) -> int` — Parse, chunk, embed a document file. Disable `do_ocr` for text-layer PDFs (biggest speedup) and `do_table_structure` if table data isn't needed. Uses `AcceleratorDevice.AUTO` (GPU when available) (async)
- `list_sources(where=None) -> list[dict]` — List distinct ingested source files with chunk counts. `where` accepts a dict (parameterised), raw SQL string, or `None` for all. Returns `{source_file, chunks_count, created_at}` (async)
- `delete(where) -> int` — Delete chunks matching a metadata filter (dict or raw SQL). Filter is required. Returns number of rows deleted (async)
- `query(query_text, top_k=5, where=None) -> list[dict]` — Cosine similarity search with optional metadata filtering. `where` accepts a dict (parameterised) or raw SQL string. Returns `{id, content, source_file, metadata, score, created_at}` (async)
