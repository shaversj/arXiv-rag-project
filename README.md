# arXiv RAG Library

This repository is a library-first foundation for Claude Agent SDK workflows
over arXiv papers.

## Supported Surface

- `arxiv_rag.query_engine` for retrieval
- `arxiv_rag.agent` for Claude-facing wrappers and tool wiring
- `arxiv_rag.tracing` for Langfuse helpers

## Commands

```bash
# Focused verification
uv run pytest tests/test_query_engine.py tests/test_agent_tools.py tests/test_tracing_langfuse.py tests/test_agent_service.py -v

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
uv run pytest tests/test_repo_surface.py tests/test_agent_tools.py tests/test_tracing_langfuse.py tests/test_agent_service.py -v
```

## Docker

```bash
# Run tests in Docker (includes PostgreSQL)
docker compose --profile test up test

# Start PostgreSQL only
docker compose up -d postgres
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

# With Langfuse tracing (add to .env)
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
uv run python -m arxiv_rag.agent.cli "What are retrieval agents?"
```