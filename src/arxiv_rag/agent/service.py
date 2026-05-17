from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from arxiv_rag.agent.citations import render_citations
from arxiv_rag.agent.models import AgentTurnResult
from arxiv_rag.agent.prompts import SYSTEM_PROMPT
from arxiv_rag.agent.tools import RetrievalTool, normalize_papers_for_tool

DEFAULT_MODEL = "claude-sonnet-4-7"
DEFAULT_ANALYSIS_LIMIT = 10
DEFAULT_SEARCH_TOOL_LIMIT = 100
ARXIV_SERVER_NAME = "arxiv"
ARXIV_SERVER_VERSION = "1.0.0"
ALLOWED_TOOLS = [
    "mcp__arxiv__search_arxiv_papers",
    "mcp__arxiv__analyze_arxiv_papers",
]
UNSUPPORTED_CORPUS_ANSWER = (
    "I couldn't answer that from the retrieved papers alone. "
    "Please try rephrasing your question or inspect the retrieved papers directly."
)
ARXIV_ID_PATTERN = re.compile(
    r"\b(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[a-z-]+)?/\d{7}(?:v\d+)?)\b",
    re.IGNORECASE,
)


def _get_model() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL)


def _tool_text_response(payload: object) -> dict[str, list[dict[str, str]]]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def _extract_user_query(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("No user message found in conversation")


def _build_default_retrieval_tool() -> RetrievalTool:
    from arxiv_rag.query_engine import QueryEngine

    query_engine = QueryEngine()
    query_engine.initialize()
    return RetrievalTool(query_engine)


def _reset_turn_tracking(retrieval_tool: object) -> None:
    reset = getattr(retrieval_tool, "reset_turn_tracking", None)
    if callable(reset):
        reset()


def _get_turn_retrieved_papers(retrieval_tool: object) -> tuple:
    getter = getattr(retrieval_tool, "get_turn_retrieved_papers", None)
    if callable(getter):
        return tuple(getter())
    return ()


def _extract_cited_ids(answer: str) -> tuple[str, ...]:
    seen: set[str] = set()
    cited_ids: list[str] = []
    for match in ARXIV_ID_PATTERN.finditer(answer):
        paper_id = match.group(0)
        if paper_id not in seen:
            cited_ids.append(paper_id)
            seen.add(paper_id)
    return tuple(cited_ids)


def _unsupported_answer_result(reason: str, extra_metadata: tuple[tuple[str, Any], ...] = ()) -> AgentTurnResult:
    return AgentTurnResult(
        answer=UNSUPPORTED_CORPUS_ANSWER,
        citations_text="",
        citations=(),
        metadata=(("grounding_status", reason),) + extra_metadata,
    )


def _ground_answer(answer: str, retrieval_tool: object) -> AgentTurnResult:
    retrieved_papers = _get_turn_retrieved_papers(retrieval_tool)
    if not retrieved_papers:
        return AgentTurnResult(
            answer=answer,
            citations_text="",
            citations=(),
        )

    papers_by_id = {}
    for paper in retrieved_papers:
        papers_by_id.setdefault(paper.id, paper)

    cited_ids = _extract_cited_ids(answer)
    if not cited_ids:
        return _unsupported_answer_result(
            "missing_citations",
            (("retrieved_paper_count", len(papers_by_id)),),
        )

    unsupported_ids = tuple(paper_id for paper_id in cited_ids if paper_id not in papers_by_id)
    if unsupported_ids:
        return _unsupported_answer_result(
            "unsupported_citations",
            (("unsupported_citation_ids", unsupported_ids),),
        )

    cited_papers = [papers_by_id[paper_id] for paper_id in cited_ids]
    citations_text, citations = render_citations(cited_papers)
    return AgentTurnResult(
        answer=answer,
        citations_text=citations_text,
        citations=citations,
        metadata=(("grounding_status", "grounded"),),
    )


def create_retrieval_server(retrieval_tool: RetrievalTool):
    @tool(
        "search_arxiv_papers",
        "Search the local arXiv paper index for relevant papers.",
        {"query": str, "limit": int},
    )
    async def search_arxiv_papers(args: Mapping[str, Any]):
        papers = retrieval_tool.search(
            args["query"],
            limit=args.get("limit", DEFAULT_SEARCH_TOOL_LIMIT),
        )
        return _tool_text_response(normalize_papers_for_tool(papers))

    @tool(
        "analyze_arxiv_papers",
        "Analyze arXiv paper metadata: count papers by author, category, or date.",
        {
            "operation": str,
            "group_by": str,
            "time_range": str,
            "query": str,
            "limit": int,
        },
    )
    async def analyze_arxiv_papers(args: Mapping[str, Any]):
        results = retrieval_tool.analyze(
            operation=args.get("operation", "count"),
            group_by=args.get("group_by", "author"),
            time_range=args.get("time_range", "all"),
            query=args.get("query"),
            limit=args.get("limit", DEFAULT_ANALYSIS_LIMIT),
        )
        return _tool_text_response(results)

    return create_sdk_mcp_server(
        name=ARXIV_SERVER_NAME,
        version=ARXIV_SERVER_VERSION,
        tools=[search_arxiv_papers, analyze_arxiv_papers],
    )


def build_agent_options(mcp_server, model: str | None = None):
    options: dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "max_turns": 3,
        "model": model or _get_model(),
        "mcp_servers": {ARXIV_SERVER_NAME: mcp_server},
        "allowed_tools": ALLOWED_TOOLS,
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        options["env"] = {"ANTHROPIC_BASE_URL": base_url}
    return ClaudeAgentOptions(**options)


class ClaudeAgentRunner:
    def __init__(self, model: str | None = None, mcp_server=None):
        self.model = model or _get_model()
        self.mcp_server = mcp_server

    async def run(self, messages: Sequence[Mapping[str, Any]]) -> str:
        prompt = _extract_user_query(messages)
        options = build_agent_options(mcp_server=self.mcp_server, model=self.model)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            chunks: list[str] = []
            async for event in client.receive_response():
                if not hasattr(event, "content"):
                    continue
                for block in event.content:
                    text = getattr(block, "text", None)
                    if text:
                        chunks.append(text)
            return "".join(chunks).strip()


async def run_agent_turn(messages, retrieval_tool=None, claude_runner=None):
    retrieval = retrieval_tool or _build_default_retrieval_tool()
    _reset_turn_tracking(retrieval)
    runner = claude_runner or ClaudeAgentRunner(
        mcp_server=create_retrieval_server(retrieval)
    )
    answer = await runner.run(messages)
    return _ground_answer(answer, retrieval)
