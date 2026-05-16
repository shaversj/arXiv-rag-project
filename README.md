# arXiv RAG Library

This repository is a library-first foundation for Claude Agent SDK workflows
over arXiv papers.

## Supported Surface

- `arxiv_rag.query_engine` for retrieval
- `arxiv_rag.agent` for Claude-facing wrappers and tool wiring

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

The evaluation dataset is a JSON array of objects with:
- `query`: the search query to run
- `relevant_ids`: a list of paper IDs that should count as relevant

Saved benchmark runs include:
- `summary.md`
- `summary.json`
- `per_query.csv`
