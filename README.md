# arXiv RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline for searching arXiv papers using semantic similarity with PostgreSQL + pgvector.

## Overview

This project ingests arXiv metadata and provides a chat interface for semantic search. It uses:

- **PostgreSQL + pgvector** - metadata storage, embeddings, full-text search, and similarity search
- **FastAPI** - REST API server
- **Sentence Transformers** - `all-MiniLM-L6-v2` for embeddings
- **Vanilla HTML/JS** - chat interface

## Quick Start

### 1. Start Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL 15 with pgvector extension (port 5432)
- FastAPI server (port 8000)

### 2. Ingest Papers

```bash
docker-compose run --rm app uv run python3 -m arxiv_rag.ingest
```

This will:
- Read `arxiv-metadata-oai-snapshot.json` (5.3GB)
- Filter for `cs.AI` category (~176k papers)
- Generate embeddings using `all-MiniLM-L6-v2`
- Store metadata + embeddings in PostgreSQL

**Time:** ~45-60 minutes on CPU (first run only)

### 3. Open Chat Interface

Open http://localhost:8000 in your browser.

## Project Structure

```
├── config.yaml              # PostgreSQL connection settings
├── pyproject.toml           # Python dependencies (uv)
├── Dockerfile               # App container
├── docker-compose.yaml       # PostgreSQL + app services
├── src/arxiv_rag/
│   └── ingest.py            # PostgresStore + ingest function
├── query_engine.py          # pgvector similarity search
├── app.py                   # FastAPI server
├── static/
│   └── index.html          # Chat UI
└── data/
    └── test_sample.json    # Sample for testing
```

## Configuration

Edit `config.yaml` to adjust:

```yaml
json_file: "arxiv-metadata-oai-snapshot.json"
category_filter: "cs.AI"           # Filter by arXiv category
db_host: "localhost"              # PostgreSQL host
db_port: 5432                     # PostgreSQL port
db_name: "arxiv_rag"              # Database name
db_user: "postgres"               # Database user
db_password: "postgres"          # Database password
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim: 384               # Embedding dimension
batch_size: 1000                 # Batch size for ingest
top_k: 10                        # Number of results to return
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/health` | GET | Health check |
| `/api/search` | POST | Search papers `{"query": "...", "filters": {}}` |
| `/api/paper/{paper_id}` | GET | Get paper by ID |

### Search Example

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning for robotics"}'
```

Response:
```json
{
  "results": [
    {
      "id": "2301.12345",
      "title": "Paper Title",
      "authors": "Author Name",
      "abstract": "Paper abstract...",
      "categories": "cs.AI cs.LG",
      "score": 0.85
    }
  ]
}
```

## Development

### Run Tests

```bash
docker-compose run --rm app pytest tests/ -v
```

### Local Development (without Docker)

```bash
# Install dependencies
uv sync

# Start PostgreSQL (with pgvector)
docker run -d --name arxiv-postgres -e POSTGRES_DB=arxiv_rag -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg15

# Run ingest
uv run python3 -m arxiv_rag.ingest

# Start server
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Data Files

The project expects `arxiv-metadata-oai-snapshot.json` in the project root. This file contains one JSON record per line with fields:

- `id` - arXiv ID (e.g., "0704.0001")
- `title` - Paper title
- `authors` - Author string
- `abstract` - Paper abstract
- `categories` - Space-separated categories (e.g., "cs.AI cs.LG")
- `update_date` - Last update date

## Performance Notes

- **176k papers** ~45-60 min to ingest on CPU
- **FAISS index not needed** - pgvector handles similarity search efficiently
- **IVF index** created on embeddings for fast lookups
- **GIN index** on full-text for keyword search

## Troubleshooting

### PostgreSQL not starting

```bash
docker-compose logs postgres
```

### Connection refused

```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart
```

### Ingest hanging

Check that the JSON file exists and is readable:
```bash
ls -la arxiv-metadata-oai-snapshot.json
head -1 arxiv-metadata-oai-snapshot.json
```