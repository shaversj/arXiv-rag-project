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
uv sync
uv run pytest tests/test_query_engine.py tests/test_agent_tools.py -q
```