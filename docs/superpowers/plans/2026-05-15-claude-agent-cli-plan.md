# Claude Agent CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current FastAPI/browser interaction path with a reusable Claude Agent SDK library entrypoint, a thin CLI chat interface, and optional Langfuse tracing.

**Architecture:** Keep retrieval in the Python package, add a focused agent layer with one `search_arxiv_papers` tool, and expose the same orchestration through a library API plus a REPL-style CLI. Remove the old HTTP and static chat surface from the supported path while keeping tests centered on retrieval, agent orchestration, citations, and fail-open tracing.

**Tech Stack:** Python 3.11+, uv, Claude Agent SDK for Python, Langfuse Python SDK, PostgreSQL + pgvector, sentence-transformers, pytest

---

## File Structure

```text
/Users/wu36/Code/arXiv-rag-project/
├── config.yaml
├── pyproject.toml
├── README.md
├── src/arxiv_rag/
│   ├── __init__.py
│   ├── ingest.py
│   ├── query_engine.py
│   ├── chat_cli.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── prompts.py
│   │   ├── citations.py
│   │   ├── tools.py
│   │   └── service.py
│   └── tracing/
│       ├── __init__.py
│       └── langfuse.py
├── tests/
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_query_engine.py
│   ├── test_agent_citations.py
│   ├── test_agent_tools.py
│   ├── test_agent_service.py
│   ├── test_chat_cli.py
│   └── test_tracing_langfuse.py
└── docs/superpowers/plans/2026-05-15-claude-agent-cli-plan.md
```

## Task 1: Rework Packaging and Remove Web Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Delete: `app.py`
- Delete: `static/index.html`
- Delete: `tests/test_app.py`

- [ ] **Step 1: Write the failing packaging regression test**

```python
from pathlib import Path


def test_web_entrypoints_removed_from_repo_surface():
    assert not Path("app.py").exists()
    assert not Path("static/index.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repo_surface.py::test_web_entrypoints_removed_from_repo_surface -v`
Expected: FAIL because `tests/test_repo_surface.py` does not exist yet.

- [ ] **Step 3: Add the temporary failing test file**

```python
from pathlib import Path


def test_web_entrypoints_removed_from_repo_surface():
    assert not Path("app.py").exists()
    assert not Path("static/index.html").exists()
```

- [ ] **Step 4: Update dependencies and project metadata**

```toml
[project]
name = "arxiv-rag"
version = "0.1.0"
description = "arXiv retrieval agent with Claude Agent SDK and CLI chat"
requires-python = ">=3.11"
dependencies = [
    "claude-agent-sdk>=0.0.20",
    "langfuse>=2.60.0",
    "psycopg2-binary>=2.9.0",
    "sentence-transformers>=2.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.60",
    "numpy>=1.21",
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
]

[project.scripts]
arxiv-rag-chat = "arxiv_rag.chat_cli:main"
```

- [ ] **Step 5: Remove the old web files and add the repo-surface test**

```bash
rm /Users/wu36/Code/arXiv-rag-project/app.py
rm /Users/wu36/Code/arXiv-rag-project/static/index.html
rm /Users/wu36/Code/arXiv-rag-project/tests/test_app.py
```

```python
from pathlib import Path


def test_web_entrypoints_removed_from_repo_surface():
    assert not Path("app.py").exists()
    assert not Path("static/index.html").exists()
```

- [ ] **Step 6: Update README usage to the CLI/library direction**

```md
## Usage

### Interactive chat

```bash
uv run arxiv-rag-chat
```

### Python API

```python
import asyncio
from arxiv_rag.agent.service import run_agent_turn


async def main():
    result = await run_agent_turn(
        [{"role": "user", "content": "What are recent themes in agentic retrieval?"}]
    )
    print(result.answer)


asyncio.run(main())
```
```

- [ ] **Step 7: Run tests to verify the packaging changes pass**

Run: `pytest tests/test_repo_surface.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md tests/test_repo_surface.py
git add -u
git commit -m "refactor: remove web interface and add CLI package entrypoint"
```

## Task 2: Move QueryEngine Into the Package and Preserve Retrieval Behavior

**Files:**
- Create: `src/arxiv_rag/query_engine.py`
- Modify: `src/arxiv_rag/__init__.py`
- Delete: `query_engine.py`
- Modify: `tests/test_query_engine.py`

- [ ] **Step 1: Write the failing import-path test**

```python
from arxiv_rag.query_engine import QueryEngine


def test_query_engine_is_importable_from_package():
    qe = QueryEngine
    assert qe is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_engine.py::test_query_engine_is_importable_from_package -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_rag.query_engine'`

- [ ] **Step 3: Copy the current implementation into the package**

```python
import json

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


class QueryEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as handle:
            self.config = yaml.safe_load(handle)
        self.store = None
        self.model = None

    def initialize(self) -> None:
        from arxiv_rag.ingest import PostgresStore

        self.store = PostgresStore(
            host=self.config.get("db_host"),
            port=self.config.get("db_port"),
            dbname=self.config.get("db_name"),
            user=self.config.get("db_user"),
            password=self.config.get("db_password"),
        )
        self.store.init_schema()
        self.model = SentenceTransformer(self.config["embedding_model"])

    def search(self, query, filters=None, limit=None):
        if not self.store:
            raise RuntimeError("QueryEngine not initialized. Call initialize() first.")
        if limit is None:
            limit = self.config["top_k"]

        query_embedding = self.model.encode(query).astype("float32")
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_vector = json.dumps(query_norm.tolist())

        conn = self.store.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.title, p.authors, p.abstract, p.categories,
                   1 - (e.embedding <=> %s::vector) AS score
            FROM papers p
            JOIN paper_embeddings e ON p.id = e.paper_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, limit * 2),
        )
        rows = cursor.fetchall()
        cursor.close()

        results = []
        seen_ids = set()
        for row in rows:
            if row[0] in seen_ids:
                continue
            seen_ids.add(row[0])
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "authors": row[2],
                    "abstract": row[3],
                    "categories": row[4],
                    "score": float(row[5]),
                }
            )
        return results[:limit]
```

- [ ] **Step 4: Update the test imports and keep the existing init guard**

```python
import pytest

from arxiv_rag.query_engine import QueryEngine


def test_query_engine_is_importable_from_package():
    qe = QueryEngine
    assert qe is not None


def test_search_requires_init():
    qe = QueryEngine()
    with pytest.raises(RuntimeError):
        qe.search("test query")
```

- [ ] **Step 5: Export the query engine from the package root**

```python
from arxiv_rag.query_engine import QueryEngine

__all__ = ["QueryEngine"]
```

- [ ] **Step 6: Remove the old top-level module**

```bash
rm /Users/wu36/Code/arXiv-rag-project/query_engine.py
```

- [ ] **Step 7: Run tests to verify retrieval import stability**

Run: `pytest tests/test_query_engine.py -v`
Expected: PASS for unit tests that do not require live data.

- [ ] **Step 8: Commit**

```bash
git add src/arxiv_rag/__init__.py src/arxiv_rag/query_engine.py tests/test_query_engine.py
git add -u
git commit -m "refactor: package query engine for agent reuse"
```

## Task 3: Add Shared Agent Models and Citation Formatting

**Files:**
- Create: `src/arxiv_rag/agent/__init__.py`
- Create: `src/arxiv_rag/agent/models.py`
- Create: `src/arxiv_rag/agent/citations.py`
- Create: `tests/test_agent_citations.py`

- [ ] **Step 1: Write the failing citation formatting test**

```python
from arxiv_rag.agent.citations import build_citation_section
from arxiv_rag.agent.models import RetrievedPaper


def test_build_citation_section_numbers_unique_papers():
    papers = [
        RetrievedPaper(
            id="1234.5678",
            title="Agent Retrieval",
            authors="Ada Lovelace",
            abstract="A",
            categories="cs.AI",
            score=0.91,
        ),
        RetrievedPaper(
            id="1234.5678",
            title="Agent Retrieval",
            authors="Ada Lovelace",
            abstract="A",
            categories="cs.AI",
            score=0.91,
        ),
    ]

    rendered, citations = build_citation_section(papers)

    assert "[1] Agent Retrieval" in rendered
    assert len(citations) == 1
    assert citations[0].url == "https://arxiv.org/abs/1234.5678"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_citations.py::test_build_citation_section_numbers_unique_papers -v`
Expected: FAIL with missing module imports.

- [ ] **Step 3: Add shared dataclasses for papers, citations, and turn results**

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievedPaper:
    id: str
    title: str
    authors: str
    abstract: str
    categories: str
    score: float


@dataclass(slots=True)
class Citation:
    index: int
    paper_id: str
    title: str
    authors: str
    url: str


@dataclass(slots=True)
class AgentTurnResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    papers: list[RetrievedPaper] = field(default_factory=list)
    trace_id: str | None = None
```

- [ ] **Step 4: Implement deterministic citation rendering**

```python
from arxiv_rag.agent.models import Citation, RetrievedPaper


def build_citation_section(papers: list[RetrievedPaper]) -> tuple[str, list[Citation]]:
    unique = []
    seen = set()
    for paper in papers:
        if paper.id in seen:
            continue
        seen.add(paper.id)
        unique.append(paper)

    citations = [
        Citation(
            index=index,
            paper_id=paper.id,
            title=paper.title,
            authors=paper.authors,
            url=f"https://arxiv.org/abs/{paper.id}",
        )
        for index, paper in enumerate(unique, start=1)
    ]
    rendered = "\n".join(
        f"[{item.index}] {item.title} — {item.authors} — {item.url}"
        for item in citations
    )
    return rendered, citations
```

- [ ] **Step 5: Export the shared agent models**

```python
from arxiv_rag.agent.models import AgentTurnResult, Citation, RetrievedPaper

__all__ = ["AgentTurnResult", "Citation", "RetrievedPaper"]
```

- [ ] **Step 6: Run tests to verify citation behavior**

Run: `pytest tests/test_agent_citations.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/arxiv_rag/agent/__init__.py src/arxiv_rag/agent/models.py src/arxiv_rag/agent/citations.py tests/test_agent_citations.py
git commit -m "feat: add agent result models and citation formatting"
```

## Task 4: Add the Retrieval Tool Adapter

**Files:**
- Create: `src/arxiv_rag/agent/tools.py`
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: Write the failing tool-shaping test**

```python
from arxiv_rag.agent.models import RetrievedPaper
from arxiv_rag.agent.tools import normalize_papers_for_tool


def test_normalize_papers_for_tool_truncates_abstracts():
    papers = [
        RetrievedPaper(
            id="1",
            title="A",
            authors="B",
            abstract="x" * 400,
            categories="cs.AI",
            score=0.5,
        )
    ]

    payload = normalize_papers_for_tool(papers)

    assert payload[0]["id"] == "1"
    assert len(payload[0]["abstract"]) <= 243
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_tools.py::test_normalize_papers_for_tool_truncates_abstracts -v`
Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement result normalization and the tool adapter class**

```python
from arxiv_rag.agent.models import RetrievedPaper
from arxiv_rag.query_engine import QueryEngine


def normalize_papers_for_tool(papers: list[RetrievedPaper]) -> list[dict]:
    normalized = []
    for paper in papers:
        abstract = paper.abstract or ""
        snippet = abstract[:240] + "..." if len(abstract) > 240 else abstract
        normalized.append(
            {
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": snippet,
                "categories": paper.categories,
                "score": paper.score,
            }
        )
    return normalized


class RetrievalTool:
    def __init__(self, query_engine: QueryEngine):
        self.query_engine = query_engine

    def search(self, query: str, limit: int = 5) -> list[RetrievedPaper]:
        rows = self.query_engine.search(query, limit=limit)
        return [
            RetrievedPaper(
                id=row["id"],
                title=row["title"],
                authors=row["authors"],
                abstract=row["abstract"],
                categories=row["categories"],
                score=row["score"],
            )
            for row in rows
        ]
```

- [ ] **Step 4: Add a test for mapping QueryEngine rows into RetrievedPaper**

```python
from arxiv_rag.agent.tools import RetrievalTool


class StubQueryEngine:
    def search(self, query, filters=None, limit=None):
        return [
            {
                "id": "42",
                "title": "Tooling Paper",
                "authors": "Grace Hopper",
                "abstract": "Interesting paper",
                "categories": "cs.AI",
                "score": 0.88,
            }
        ]


def test_retrieval_tool_maps_query_engine_rows():
    tool = RetrievalTool(StubQueryEngine())

    results = tool.search("tooling", limit=1)

    assert results[0].id == "42"
    assert results[0].title == "Tooling Paper"
```

- [ ] **Step 5: Run tests to verify tool shaping**

Run: `pytest tests/test_agent_tools.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arxiv_rag/agent/tools.py tests/test_agent_tools.py
git commit -m "feat: add retrieval tool adapter for the agent"
```

## Task 5: Add Langfuse Fail-Open Tracing Helpers

**Files:**
- Create: `src/arxiv_rag/tracing/__init__.py`
- Create: `src/arxiv_rag/tracing/langfuse.py`
- Create: `tests/test_tracing_langfuse.py`

- [ ] **Step 1: Write the failing fail-open tracing test**

```python
from arxiv_rag.tracing.langfuse import build_tracer


def test_build_tracer_returns_noop_when_env_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    tracer = build_tracer()

    assert tracer.enabled is False
    assert tracer.trace_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing_langfuse.py::test_build_tracer_returns_noop_when_env_missing -v`
Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement a small fail-open tracer abstraction**

```python
import os
from dataclasses import dataclass


@dataclass
class TraceHandle:
    enabled: bool
    trace_id: str | None = None

    def record_tool(self, *, name: str, input_payload: dict, output_payload: dict) -> None:
        return None

    def record_result(self, *, answer: str) -> None:
        return None


def build_tracer() -> TraceHandle:
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    ):
        return TraceHandle(enabled=False, trace_id=None)
    return TraceHandle(enabled=True, trace_id="configured-at-runtime")
```

- [ ] **Step 4: Add a second test for the configured path**

```python
from arxiv_rag.tracing.langfuse import build_tracer


def test_build_tracer_marks_configured_env_as_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    tracer = build_tracer()

    assert tracer.enabled is True
```

- [ ] **Step 5: Run tests to verify fail-open tracing behavior**

Run: `pytest tests/test_tracing_langfuse.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arxiv_rag/tracing/__init__.py src/arxiv_rag/tracing/langfuse.py tests/test_tracing_langfuse.py
git commit -m "feat: add fail-open Langfuse tracing helpers"
```

## Task 6: Add the Claude Agent Service Entrypoint

**Files:**
- Create: `src/arxiv_rag/agent/prompts.py`
- Create: `src/arxiv_rag/agent/service.py`
- Create: `tests/test_agent_service.py`

- [ ] **Step 1: Write the failing orchestration test**

```python
import pytest

from arxiv_rag.agent.models import RetrievedPaper
from arxiv_rag.agent.service import run_agent_turn


class StubClaudeRunner:
    async def run(self, messages, papers):
        return "Agents frequently combine retrieval with synthesis [1]."


class StubRetrievalTool:
    def search(self, query: str, limit: int = 5):
        return [
            RetrievedPaper(
                id="1111.1111",
                title="Retrieval Agents",
                authors="Jane Doe",
                abstract="Agent paper",
                categories="cs.AI",
                score=0.9,
            )
        ]


@pytest.mark.asyncio
async def test_run_agent_turn_returns_answer_citations_and_papers():
    result = await run_agent_turn(
        [{"role": "user", "content": "What do retrieval agents do?"}],
        retrieval_tool=StubRetrievalTool(),
        claude_runner=StubClaudeRunner(),
    )

    assert "retrieval" in result.answer.lower()
    assert result.citations[0].paper_id == "1111.1111"
    assert result.papers[0].title == "Retrieval Agents"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_service.py::test_run_agent_turn_returns_answer_citations_and_papers -v`
Expected: FAIL with missing module imports.

- [ ] **Step 3: Add the retrieval-only system prompt**

```python
SYSTEM_PROMPT = """You answer questions about arXiv papers.
Use the search_arxiv_papers tool when evidence is needed.
Base claims on retrieved papers only.
If retrieval is weak or empty, say so clearly.
Include citations that match the provided paper list.
Do not attempt actions outside retrieval and synthesis.
"""
```

- [ ] **Step 4: Implement the service with dependency injection**

```python
from arxiv_rag.agent.citations import build_citation_section
from arxiv_rag.agent.models import AgentTurnResult
from arxiv_rag.agent.tools import RetrievalTool
from arxiv_rag.query_engine import QueryEngine
from arxiv_rag.tracing.langfuse import build_tracer


async def run_agent_turn(messages, retrieval_tool=None, claude_runner=None):
    tracer = build_tracer()
    tool = retrieval_tool
    if tool is None:
        query_engine = QueryEngine()
        query_engine.initialize()
        tool = RetrievalTool(query_engine)

    user_query = next(
        message["content"] for message in reversed(messages) if message["role"] == "user"
    )
    papers = tool.search(user_query, limit=5)
    rendered_citations, citations = build_citation_section(papers)

    if claude_runner is None:
        answer = (
            "Claude runner integration is not wired yet. "
            "Retrieved papers are available for synthesis.\n\n"
            f"{rendered_citations}"
        )
    else:
        answer = await claude_runner.run(messages, papers)

    tracer.record_result(answer=answer)
    return AgentTurnResult(
        answer=answer,
        citations=citations,
        papers=papers,
        trace_id=tracer.trace_id,
    )
```

- [ ] **Step 5: Add an empty-retrieval test**

```python
import pytest

from arxiv_rag.agent.service import run_agent_turn


class EmptyRetrievalTool:
    def search(self, query: str, limit: int = 5):
        return []


class EmptyClaudeRunner:
    async def run(self, messages, papers):
        return "No relevant papers were found."


@pytest.mark.asyncio
async def test_run_agent_turn_handles_empty_retrieval():
    result = await run_agent_turn(
        [{"role": "user", "content": "query"}],
        retrieval_tool=EmptyRetrievalTool(),
        claude_runner=EmptyClaudeRunner(),
    )

    assert result.papers == []
    assert result.citations == []
```

- [ ] **Step 6: Replace the placeholder runner with Claude Agent SDK integration**

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


class ClaudeAgentRunner:
    async def run(self, messages, papers):
        prompt = messages[-1]["content"]
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=3,
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            chunks = []
            async for event in client.receive_response():
                if hasattr(event, "content"):
                    for block in event.content:
                        text = getattr(block, "text", None)
                        if text:
                            chunks.append(text)
            return "".join(chunks).strip()
```

- [ ] **Step 7: Run tests to verify service behavior**

Run: `pytest tests/test_agent_service.py -v`
Expected: PASS with injected test doubles.

- [ ] **Step 8: Commit**

```bash
git add src/arxiv_rag/agent/prompts.py src/arxiv_rag/agent/service.py tests/test_agent_service.py
git commit -m "feat: add Claude agent service entrypoint"
```

## Task 7: Register the SDK Tool and Trace Tool Use

**Files:**
- Modify: `src/arxiv_rag/agent/tools.py`
- Modify: `src/arxiv_rag/agent/service.py`
- Modify: `src/arxiv_rag/tracing/langfuse.py`
- Modify: `tests/test_agent_service.py`

- [ ] **Step 1: Write the failing tool-registration test**

```python
from arxiv_rag.agent.service import build_agent_options


def test_build_agent_options_allows_only_search_tool():
    options = build_agent_options(mcp_server="server")

    assert options.allowed_tools == ["mcp__arxiv__search_arxiv_papers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_service.py::test_build_agent_options_allows_only_search_tool -v`
Expected: FAIL because `build_agent_options` does not exist.

- [ ] **Step 3: Expose the Claude SDK tool using the SDK MCP helper**

```python
from claude_agent_sdk import create_sdk_mcp_server, tool


def create_retrieval_server(retrieval_tool):
    @tool(
        "search_arxiv_papers",
        "Search the local arXiv paper index for relevant papers.",
        {"query": str, "limit": int},
    )
    async def search_arxiv_papers(args):
        papers = retrieval_tool.search(args["query"], limit=args.get("limit", 5))
        return {"content": [{"type": "text", "text": str(normalize_papers_for_tool(papers))}]}

    return create_sdk_mcp_server(
        name="arxiv",
        version="1.0.0",
        tools=[search_arxiv_papers],
    )
```

- [ ] **Step 4: Build Claude agent options around the single allowed tool**

```python
from claude_agent_sdk import ClaudeAgentOptions


def build_agent_options(mcp_server):
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        max_turns=3,
        mcp_servers={"arxiv": mcp_server},
        allowed_tools=["mcp__arxiv__search_arxiv_papers"],
    )
```

- [ ] **Step 5: Record tool payloads in the tracer**

```python
@dataclass
class TraceHandle:
    enabled: bool
    trace_id: str | None = None
    tool_events: list[dict] | None = None

    def record_tool(self, *, name: str, input_payload: dict, output_payload: dict) -> None:
        if self.tool_events is None:
            self.tool_events = []
        self.tool_events.append(
            {
                "name": name,
                "input": input_payload,
                "output": output_payload,
            }
        )
```

- [ ] **Step 6: Run tests to verify the single-tool contract**

Run: `pytest tests/test_agent_service.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/arxiv_rag/agent/tools.py src/arxiv_rag/agent/service.py src/arxiv_rag/tracing/langfuse.py tests/test_agent_service.py
git commit -m "feat: register Claude SDK retrieval tool"
```

## Task 8: Add the Interactive CLI

**Files:**
- Create: `src/arxiv_rag/chat_cli.py`
- Create: `tests/test_chat_cli.py`

- [ ] **Step 1: Write the failing CLI render test**

```python
from arxiv_rag.chat_cli import render_turn
from arxiv_rag.agent.models import AgentTurnResult, Citation


def test_render_turn_includes_answer_and_citations():
    turn = AgentTurnResult(
        answer="Answer body",
        citations=[
            Citation(
                index=1,
                paper_id="1234.5678",
                title="Paper",
                authors="Author",
                url="https://arxiv.org/abs/1234.5678",
            )
        ],
    )

    rendered = render_turn(turn)

    assert "Answer body" in rendered
    assert "[1] Paper" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_cli.py::test_render_turn_includes_answer_and_citations -v`
Expected: FAIL with missing module imports.

- [ ] **Step 3: Implement rendering and the interactive loop**

```python
import asyncio

from arxiv_rag.agent.citations import build_citation_section
from arxiv_rag.agent.service import run_agent_turn


def render_turn(turn):
    lines = [turn.answer]
    if turn.citations:
        lines.append("")
        lines.append("Citations:")
        for citation in turn.citations:
            lines.append(f"[{citation.index}] {citation.title} — {citation.url}")
    return "\n".join(lines)


async def chat_loop():
    messages = []
    while True:
        user_input = input("You> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        turn = await run_agent_turn(messages)
        print(render_turn(turn))
        messages.append({"role": "assistant", "content": turn.answer})


def main():
    asyncio.run(chat_loop())
```

- [ ] **Step 4: Add an exit-path test**

```python
def test_render_turn_includes_answer_and_citations():
    turn = AgentTurnResult(
        answer="Answer body",
        citations=[
            Citation(
                index=1,
                paper_id="1234.5678",
                title="Paper",
                authors="Author",
                url="https://arxiv.org/abs/1234.5678",
            )
        ],
    )

    rendered = render_turn(turn)

    assert "Answer body" in rendered
    assert "[1] Paper" in rendered


def test_render_turn_without_citations_returns_answer_only():
    turn = AgentTurnResult(answer="Just text")

    rendered = render_turn(turn)

    assert rendered == "Just text"
```

- [ ] **Step 5: Run tests to verify CLI behavior**

Run: `pytest tests/test_chat_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arxiv_rag/chat_cli.py tests/test_chat_cli.py
git commit -m "feat: add interactive CLI chat interface"
```

## Task 9: Wire Real Langfuse Client Usage

**Files:**
- Modify: `src/arxiv_rag/tracing/langfuse.py`
- Modify: `src/arxiv_rag/agent/service.py`
- Modify: `tests/test_tracing_langfuse.py`

- [ ] **Step 1: Write the failing configured-trace test**

```python
from arxiv_rag.tracing.langfuse import TraceHandle


def test_trace_handle_records_result_text():
    trace = TraceHandle(enabled=True, trace_id="trace-123", tool_events=[])

    trace.record_result(answer="hello")

    assert trace.answer == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing_langfuse.py::test_trace_handle_records_result_text -v`
Expected: FAIL because `TraceHandle.answer` is not implemented.

- [ ] **Step 3: Expand the tracer to keep answer state and wrap the Langfuse client**

```python
from dataclasses import dataclass, field

from langfuse import get_client


@dataclass
class TraceHandle:
    enabled: bool
    trace_id: str | None = None
    tool_events: list[dict] = field(default_factory=list)
    answer: str | None = None

    def record_result(self, *, answer: str) -> None:
        self.answer = answer


def build_tracer() -> TraceHandle:
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    ):
        return TraceHandle(enabled=False, trace_id=None)
    client = get_client()
    trace = client.trace(name="arxiv-agent-turn")
    return TraceHandle(enabled=True, trace_id=trace.id)
```

- [ ] **Step 4: Update the agent service to record retrieval and answer events**

```python
tracer.record_tool(
    name="search_arxiv_papers",
    input_payload={"query": user_query, "limit": 5},
    output_payload={"papers": normalize_papers_for_tool(papers)},
)
tracer.record_result(answer=answer)
```

- [ ] **Step 5: Run tests to verify trace bookkeeping**

Run: `pytest tests/test_tracing_langfuse.py tests/test_agent_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arxiv_rag/tracing/langfuse.py src/arxiv_rag/agent/service.py tests/test_tracing_langfuse.py
git commit -m "feat: record agent tool and answer trace events"
```

## Task 10: Final Verification and Documentation Sweep

**Files:**
- Modify: `README.md`
- Modify: `config.yaml`
- Modify: `docs/superpowers/specs/2026-05-15-claude-agent-cli-design.md`

- [ ] **Step 1: Add the final README commands**

```md
## Commands

```bash
# Run the CLI
uv run arxiv-rag-chat

# Run focused tests
pytest tests/test_query_engine.py tests/test_agent_tools.py tests/test_agent_service.py -v

# Run the full suite
pytest tests/ -v
```
```

- [ ] **Step 2: Add config keys for the agent path**

```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
top_k: 10
claude_model: "sonnet"
agent_max_turns: 3
agent_search_limit: 5
```

- [ ] **Step 3: Run the full verification suite**

Run: `pytest tests/ -v`
Expected: PASS for all unit and integration-style tests that use stubs instead of live model calls.

- [ ] **Step 4: Run a CLI smoke check**

Run: `printf 'quit\n' | uv run arxiv-rag-chat`
Expected: output contains `Bye.`

- [ ] **Step 5: Commit**

```bash
git add README.md config.yaml docs/superpowers/specs/2026-05-15-claude-agent-cli-design.md
git add docs/superpowers/plans/2026-05-15-claude-agent-cli-plan.md
git commit -m "docs: finalize Claude agent CLI migration plan"
```
