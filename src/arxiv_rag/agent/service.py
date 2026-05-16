from __future__ import annotations

import os

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from arxiv_rag.agent.citations import render_citations
from arxiv_rag.agent.models import AgentTurnResult
from arxiv_rag.agent.prompts import SYSTEM_PROMPT
from arxiv_rag.agent.tools import RetrievalTool, normalize_papers_for_tool


def _get_model() -> str:
    return os.getenv("MODEL_NAME", "claude-sonnet-4-7")


def create_retrieval_server(retrieval_tool):
    @tool(
        "search_arxiv_papers",
        "Search the local arXiv paper index for relevant papers.",
        {"query": str, "limit": int},
    )
    async def search_arxiv_papers(args):
        papers = retrieval_tool.search(args["query"], limit=args.get("limit", 100))
        return {
            "content": [
                {"type": "text", "text": str(normalize_papers_for_tool(papers))}
            ]
        }

    @tool(
        "analyze_arxiv_papers",
        "Analyze arXiv paper metadata: count papers by author, category, or date.",
        {
            "operation": str,  # "count"
            "group_by": str,   # "author", "category", "date"
            "time_range": str,  # "7d", "30d", "90d", "all"
            "query": str,       # optional text filter
            "limit": int,
        },
    )
    async def analyze_arxiv_papers(args):
        results = retrieval_tool.analyze(
            operation=args.get("operation", "count"),
            group_by=args.get("group_by", "author"),
            time_range=args.get("time_range", "30d"),
            query=args.get("query"),
            limit=args.get("limit", 10),
        )
        return {
            "content": [
                {"type": "text", "text": str(results)}
            ]
        }

    return create_sdk_mcp_server(name="arxiv", version="1.0.0", tools=[search_arxiv_papers, analyze_arxiv_papers])


def build_agent_options(mcp_server, model: str | None = None):
    opts = {
        "system_prompt": SYSTEM_PROMPT,
        "max_turns": 3,
        "model": model or _get_model(),
        "mcp_servers": {"arxiv": mcp_server},
        "allowed_tools": ["mcp__arxiv__search_arxiv_papers", "mcp__arxiv__analyze_arxiv_papers"],
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        opts["env"] = {"ANTHROPIC_BASE_URL": base_url}
    return ClaudeAgentOptions(**opts)


class ClaudeAgentRunner:
    def __init__(self, model: str | None = None, mcp_server=None):
        self.model = model or _get_model()
        self.mcp_server = mcp_server

    async def run(self, messages, papers):
        user_query = messages[-1]["content"]
        # Build context from retrieved papers
        if papers:
            context_lines = ["Here are relevant papers from the arXiv search:"]
            for i, paper in enumerate(papers, 1):
                context_lines.append(f"\n{i}. {paper.id}: {paper.title}")
                context_lines.append(f"   Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}")
                context_lines.append(f"   Abstract: {paper.abstract[:300]}..." if len(paper.abstract) > 300 else f"   Abstract: {paper.abstract}")
            context = "\n".join(context_lines)
            prompt = f"{context}\n\nUser question: {user_query}"
        else:
            prompt = user_query

        opts = build_agent_options(mcp_server=self.mcp_server, model=self.model)
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(prompt)
            chunks = []
            async for event in client.receive_response():
                if hasattr(event, "content"):
                    for block in event.content:
                        text = getattr(block, "text", None)
                        if text:
                            chunks.append(text)
            return "".join(chunks).strip()


async def run_agent_turn(messages, retrieval_tool=None, claude_runner=None):
    retrieval = retrieval_tool
    if retrieval is None:
        from arxiv_rag.query_engine import QueryEngine

        query_engine = QueryEngine()
        query_engine.initialize()
        retrieval = RetrievalTool(query_engine)

    user_query = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        None
    )
    if user_query is None:
        raise ValueError("No user message found in conversation")
    limit = 5
    papers = retrieval.search(user_query, limit=limit)
    citations_text, citations = render_citations(papers)

    runner = claude_runner or ClaudeAgentRunner(
        mcp_server=create_retrieval_server(retrieval) if retrieval else None
    )
    answer = await runner.run(messages, papers)

    return AgentTurnResult(
        answer=answer,
        citations_text=citations_text,
        citations=citations,
    )
