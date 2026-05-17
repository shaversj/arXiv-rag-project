# arXiv RAG Library

This repository is a library-first foundation for Claude Agent SDK workflows
over arXiv papers.

## Supported Surface

- `arxiv_rag.query_engine` for retrieval
- `arxiv_rag.agent` for Claude-facing wrappers and tool wiring

The project also uses Langfuse for observability and Claude Agent SDK tracing.

The default retrieval mode is hybrid:
- semantic retrieval from pgvector embeddings
- keyword retrieval from PostgreSQL full-text search
- rank-based fusion of both result lists

## Commands

```bash
# Focused verification
uv run pytest tests/test_query_engine.py tests/test_agent_tools.py tests/test_agent_service.py -v

# Full verification
uv run pytest tests/ -v
```

## Quick Start

```bash
# Start PostgreSQL
docker compose up -d postgres

# Install dependencies
uv sync

# Run tests (requires PostgreSQL)
uv run pytest tests/test_repo_surface.py tests/test_agent_tools.py tests/test_agent_service.py -v
```

## Ingest Data

The ingester reads newline-delimited arXiv metadata from the file configured in
`config.yaml`, filters by `category_filter`, embeds each paper with the configured
sentence-transformer model, and writes both metadata and embeddings into PostgreSQL.

The expected source file is the Kaggle arXiv metadata snapshot named
`arxiv-metadata-oai-snapshot.json`.

Before running ingest:
- make sure PostgreSQL is running
- make sure `config.yaml` points `json_file` at your local arXiv metadata snapshot
- update `category_filter` if you want a category other than `cs.AI`

```bash
# Start PostgreSQL
docker compose up -d postgres

# Install dependencies
uv sync

# Run ingest with the checked-in config
uv run python -m arxiv_rag.ingest
```

Useful config fields in [`config.yaml`](/Users/wu36/Code/arXiv-rag-project/config.yaml:1):
- `json_file`: path to the newline-delimited arXiv metadata file
- `category_filter`: category token to keep during ingest
- `batch_size`: number of papers to accumulate before flushing inserts
- `embedding_model`: embedding model used during ingest and retrieval

The current project notes expect roughly 45-60 minutes for a full CPU ingest of
the `cs.AI` slice.

## Docker

```bash
# Run tests in Docker (includes PostgreSQL)
docker compose --profile test up test

# Start PostgreSQL only
docker compose up -d postgres

# Tail PostgreSQL query logs
docker compose logs -f postgres
```

## Run the Agent

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start PostgreSQL
docker compose up -d postgres

# Ask a question
uv run python -m arxiv_rag.agent.cli "What are retrieval agents?"
```

Agent runs initialize Langfuse observability from `.env` before starting the CLI.

## Evaluate Retrieval

```bash
# Run the retrieval benchmark against a labeled query set
uv run python -m arxiv_rag.evaluate data/eval_queries.json

# Save reusable benchmark artifacts for later comparison
uv run python -m arxiv_rag.evaluate data/eval_queries.json --output-dir eval_results/

# Inspect one query manually against expected paper IDs
uv run arxiv-rag-inspect-query \
  "category learning and human categorization" \
  --expected-id 2403.03835 \
  --expected-id 1304.3432
```

By default these commands use the checked-in `config.yaml`, which currently sets
`retrieval_mode: "hybrid"`.

The evaluation dataset is a JSON array of objects with:
- `query`: the search query to run
- `relevant_ids`: a list of paper IDs that should count as relevant

Saved benchmark runs include:
- `summary.md`
- `summary.json`
- `per_query.csv`
