from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest

from arxiv_rag.agent.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from arxiv_rag.query_engine import QueryEngine


def test_system_prompt_mentions_retrieval_and_citations():
    assert "search_arxiv_papers" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()


def test_build_agent_options_allows_agent_tools():
    from arxiv_rag.agent.service import build_agent_options

    options = build_agent_options(mcp_server="server")

    assert options.allowed_tools == [
        "mcp__arxiv__search_arxiv_papers",
        "mcp__arxiv__analyze_arxiv_papers",
    ]


@pytest.mark.asyncio
async def test_analyze_tool_defaults_to_full_corpus(monkeypatch):
    from arxiv_rag.agent import service

    captured: dict[str, object] = {}

    class StubRetrieval:
        def analyze(self, **kwargs):
            captured.update(kwargs)
            return [{"category": "cs.AI", "paper_count": 10}]

    def identity_tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    monkeypatch.setattr(service, "tool", identity_tool)
    monkeypatch.setattr(
        service,
        "create_sdk_mcp_server",
        lambda name, version, tools: {"tools": tools},
    )

    server = service.create_retrieval_server(StubRetrieval())
    analyze_tool = next(tool for tool in server["tools"] if tool.__name__ == "analyze_arxiv_papers")

    result = await analyze_tool({})

    assert captured["time_range"] == "all"
    assert result == {"content": [{"type": "text", "text": "[{'category': 'cs.AI', 'paper_count': 10}]"}]}


# === Tests for run_agent_turn ===

from arxiv_rag.agent.service import run_agent_turn


class StubRetrievalTool:
    def search(self, query: str, limit: int = 5):
        raise AssertionError("run_agent_turn should rely on MCP tools instead of eager search")


class StubClaudeRunner:
    async def run(self, messages):
        assert messages == [{"role": "user", "content": "What do retrieval agents do?"}]
        return "Grounded answer."


@pytest.mark.asyncio
async def test_run_agent_turn_relies_on_mcp_tools_instead_of_eager_search():
    result = await run_agent_turn(
        [{"role": "user", "content": "What do retrieval agents do?"}],
        retrieval_tool=StubRetrievalTool(),
        claude_runner=StubClaudeRunner(),
    )

    assert result.answer == "Grounded answer."
    assert result.citations == ()
    assert result.citations_text == ""
    assert result.metadata == ()


class EmptyRetrievalTool:
    def search(self, query: str, limit: int = 5):
        raise AssertionError("run_agent_turn should rely on MCP tools instead of eager search")


class EmptyClaudeRunner:
    async def run(self, messages):
        return "No relevant papers were found."


@pytest.mark.asyncio
async def test_run_agent_turn_handles_empty_retrieval():
    result = await run_agent_turn(
        [{"role": "user", "content": "query"}],
        retrieval_tool=EmptyRetrievalTool(),
        claude_runner=EmptyClaudeRunner(),
    )

    assert result.citations == ()
    assert result.citations_text == ""


@pytest.mark.asyncio
async def test_cli_main_initializes_observability_before_running(monkeypatch):
    from arxiv_rag.agent import cli

    events: list[str] = []

    class StubQueryEngine:
        def initialize(self):
            events.append("query_engine.initialize")

    class StubRetrievalTool:
        def __init__(self, query_engine):
            assert isinstance(query_engine, StubQueryEngine)
            events.append("retrieval_tool.init")

    async def fake_run_agent_turn(messages, retrieval_tool):
        assert messages == [{"role": "user", "content": "What are retrieval agents?"}]
        assert isinstance(retrieval_tool, StubRetrievalTool)
        events.append("run_agent_turn")
        return types.SimpleNamespace(answer="answer", citations_text="citations")

    monkeypatch.setattr(cli, "init_observability", lambda: events.append("init_observability"))
    monkeypatch.setattr(cli, "QueryEngine", StubQueryEngine)
    monkeypatch.setattr(cli, "RetrievalTool", StubRetrievalTool)
    monkeypatch.setattr(cli, "run_agent_turn", fake_run_agent_turn)

    await cli.main("What are retrieval agents?")

    assert events == [
        "init_observability",
        "query_engine.initialize",
        "retrieval_tool.init",
        "run_agent_turn",
    ]
