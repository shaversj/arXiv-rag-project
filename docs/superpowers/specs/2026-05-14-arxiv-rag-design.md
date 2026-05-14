# arXiv RAG Pipeline Specification

## Overview

A RAG (Retrieval-Augmented Generation) pipeline that ingests arXiv metadata from a JSON file and provides a chat interface for querying papers. The system indexes `cs.AI` category papers (~176k records) from `arxiv-metadata-oai-snapshot.json`.

## Architecture

```
arxiv-metadata-oai-snapshot.json
        ↓
   [Ingest Script]
        ↓
SQLite (metadata) + ChromaDB (embeddings)
        ↓
   [FastAPI Backend]
        ↓
   [Chat UI - HTML/JS]
```

## Data Flow

1. **Ingest** — Parse JSON, store metadata in SQLite, generate embeddings with `all-MiniLM-L6-v2`, store in ChromaDB
2. **Query** — User query → embed → ChromaDB similarity search → SQLite metadata + filters → ranked results
3. **Chat** — Stream results to web UI with paper cards (title, authors, abstract snippet, metadata)

## Components

### 1. Configuration (`config.yaml`)

```yaml
json_file: "arxiv-metadata-oai-snapshot.json"
category_filter: "cs.AI"
sqlite_db: "data/arxiv.db"
chroma_db_dir: "data/chroma"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
batch_size: 1000
top_k: 10
```

### 2. Ingest Script (`ingest.py`)

- Read JSON line-by-line (5.3GB file, streaming)
- Filter for `cs.AI` category
- Parse fields: id, title, authors, abstract, categories, journal-ref, doi, submitter, update_date
- Batch insert to SQLite with FTS5 on title/authors/abstract
- Generate embedding for title + abstract (concatenated)
- Store in ChromaDB with metadata (id, title, authors)
- Progress logging every 1000 records
- Track counts: total, ingested, skipped, errors

### 3. Query Engine (`query_engine.py`)

- `search(query: str, filters: dict) -> List[dict]`
  - filters: date_range, categories, authors (optional)
  - Returns top_k results with scores
- `generate_answer(query: str, context: list) -> str` (optional, for LLM summarization)

### 4. FastAPI Server (`app.py`)

Endpoints:
- `POST /api/search` — `{"query": "...", "filters": {...}}` → results
- `GET /api/paper/{paper_id}` — full paper metadata
- `GET /health` — health check

### 5. Chat UI (`static/index.html`)

- Single-page web app
- Message history (user + assistant)
- Input field with send button
- Results displayed as paper cards:
  - Title (clickable link to arXiv)
  - Authors
  - Score / relevance
  - Abstract snippet (truncated)
  - Metadata (date, category, journal-ref)
- Loading state during search
- Error handling with user-friendly messages

## Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | `all-MiniLM-L6-v2` | Fast, good quality, CPU-friendly |
| Vector DB | ChromaDB | Simple setup, good for local use |
| Metadata DB | SQLite + FTS5 | Zero config, full-text search built-in |
| Web framework | FastAPI | Lightweight, good for this use case |
| Frontend | Vanilla HTML/JS | No build step, simple deployment |

## Ingest Estimate

- **Records:** ~176k cs.AI papers
- **Time:** ~45-60 minutes on laptop CPU
- **Storage:** ~200-300MB (SQLite + ChromaDB)

## Scope Boundaries

- Single JSON file input (not a directory)
- Filter by `cs.AI` category only
- No LLM summarization in v1 (just retrieval)
- Local-only deployment (no cloud/API)