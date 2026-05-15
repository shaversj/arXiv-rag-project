from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from arxiv_rag.agent.models import RetrievedPaper
from arxiv_rag.agent.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from arxiv_rag.query_engine import QueryEngine

_ABSTRACT_SNIPPET_LIMIT = 240


def test_system_prompt_mentions_retrieval_and_citations():
    assert "retrieval" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()


def test_build_agent_options_allows_only_search_tool():
    from arxiv_rag.agent.service import build_agent_options

    options = build_agent_options(mcp_server="server")

    assert options.allowed_tools == ["mcp__arxiv__search_arxiv_papers"]


# === New tests for run_agent_turn ===

from arxiv_rag.agent.service import run_agent_turn


class StubRetrievalTool:
    def search(self, query: str, limit: int = 5):
        return [
            RetrievedPaper(
                id="1111.1111",
                title="Retrieval Agents",
                abstract="Agent paper",
                authors=("Jane Doe",),
                categories=("cs.AI",),
            )
        ]


class StubClaudeRunner:
    async def run(self, messages, papers):
        return "Grounded answer [1]."


@pytest.mark.asyncio
async def test_run_agent_turn_returns_answer_citations_and_papers():
    result = await run_agent_turn(
        [{"role": "user", "content": "What do retrieval agents do?"}],
        retrieval_tool=StubRetrievalTool(),
        claude_runner=StubClaudeRunner(),
    )

    assert result.answer == "Grounded answer [1]."
    assert result.citations[0].id == "1111.1111"


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

    assert result.citations == ()
    assert result.citations_text == ""