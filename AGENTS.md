# AGENTS.md - arXiv RAG Pipeline

## Project Overview

This is a RAG (Retrieval-Augmented Generation) pipeline for searching arXiv papers with hybrid retrieval. It uses PostgreSQL + pgvector for metadata storage, embeddings, full-text search, and fused ranking.

## Tech Stack

- **PostgreSQL 15 + pgvector** - metadata, embeddings, full-text search, hybrid retrieval
- **Python 3.11+** - application code
- **uv** - package manager
- **Docker Compose** - services (PostgreSQL + app)
- **FastAPI** - REST API server
- **sentence-transformers** - `all-MiniLM-L6-v2` embeddings
- **Langfuse** - observability and Claude Agent SDK tracing

## Key Files

| File | Purpose |
|------|--------|
| `src/arxiv_rag/ingest.py` | PostgresStore class, ingest function |
| `query_engine.py` | QueryEngine for semantic, keyword, and hybrid retrieval |
| `app.py` | FastAPI server with /api/search endpoint |
| `config.yaml` | Database connection, model settings |
| `docker-compose.yaml` | PostgreSQL + app services |

## Important Conventions

### Package Structure
- Application code lives in `src/arxiv_rag/` (installed as package via hatchling)
- Entry point: `uv run python3 -m arxiv_rag.ingest`

### Database
- PostgreSQL with pgvector extension
- Tables: `papers` (metadata), `paper_embeddings` (vectors)
- IVFFlat index on embeddings for similarity search
- GIN index on papers for full-text search

### Configuration
- All config via `config.yaml` (no hardcoded values)
- Database credentials via env vars: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- Default retrieval mode is `hybrid`
- Hybrid retrieval combines pgvector semantic search and PostgreSQL full-text search with rank-based fusion
- Ingest reads newline-delimited arXiv metadata from `json_file` and filters rows by `category_filter`
- Langfuse credentials are loaded from `.env` when agent observability initializes

### Docker
- `docker-compose up -d` starts PostgreSQL + app
- PostgreSQL health check via `pg_isready`
- App depends on postgres being healthy

## Commands

```bash
# Start services
docker-compose up -d

# Stop (keep data)
docker-compose down

# Stop (delete data)
docker-compose down -v

# Run ingest
docker-compose run --rm app uv run python3 -m arxiv_rag.ingest

# Run tests
docker-compose run --rm app pytest tests/ -v

# Rebuild app container
docker-compose build app
```

## Ingest Notes

- Local ingest entry point: `uv run python3 -m arxiv_rag.ingest`
- The expected input file is the Kaggle arXiv metadata snapshot `arxiv-metadata-oai-snapshot.json`
- The ingester expects `config.yaml` to point `json_file` at an arXiv metadata snapshot on disk
- Only papers whose category list contains `category_filter` are embedded and stored
- Metadata is written to `papers`, embeddings to `paper_embeddings`

## Project State

- Ingest creates ~176k paper records from cs.AI category
- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Full ingestion takes ~45-60 minutes on CPU

## Adding Dependencies

If you add a new Python package:
1. Add to `pyproject.toml` under `dependencies`
2. Run `uv sync` to update lock file
3. Rebuild if running in Docker: `docker-compose build app`

## Architecture Notes

- No separate embedding store - embeddings stored in PostgreSQL via pgvector
- FAISS not used - pgvector handles semantic similarity search efficiently
- Default search behavior is hybrid fusion over semantic and keyword result lists
- Agent observability is instrumented through Langfuse in `src/arxiv_rag/observability.py`
- SQLite not used - PostgreSQL for data integrity and concurrent access
