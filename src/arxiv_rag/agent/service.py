from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from arxiv_rag.agent.citations import render_citations
from arxiv_rag.agent.models import AgentTurnResult
from arxiv_rag.agent.prompts import SYSTEM_PROMPT
from arxiv_rag.agent.tools import RetrievalTool, normalize_papers_for_tool
from arxiv_rag.tracing.langfuse import build_tracer


def create_retrieval_server(retrieval_tool):
    @tool(
        "search_arxiv_papers",
        "Search the local arXiv paper index for relevant papers.",
        {"query": str, "limit": int},
    )
    async def search_arxiv_papers(args):
        papers = retrieval_tool.search(args["query"], limit=args.get("limit", 5))
        return {
            "content": [
                {"type": "text", "text": str(normalize_papers_for_tool(papers))}
            ]
        }

    return create_sdk_mcp_server(name="arxiv", version="1.0.0", tools=[search_arxiv_papers])


def build_agent_options(mcp_server):
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        max_turns=3,
        mcp_servers={"arxiv": mcp_server},
        allowed_tools=["mcp__arxiv__search_arxiv_papers"],
    )


class ClaudeAgentRunner:
    async def run(self, messages, papers):
        prompt = messages[-1]["content"]
        async with ClaudeSDKClient(options=build_agent_options(mcp_server={})) as client:
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
    tracer = build_tracer(input_payload={"messages": messages})
    tool = retrieval_tool
    if tool is None:
        from arxiv_rag.query_engine import QueryEngine

        query_engine = QueryEngine()
        query_engine.initialize()
        tool = RetrievalTool(query_engine)

    user_query = next(
        message["content"] for message in reversed(messages) if message["role"] == "user"
    )
    papers = tool.search(user_query, limit=5)
    citations_text, citations = render_citations(papers)

    runner = claude_runner or ClaudeAgentRunner()
    answer = await runner.run(messages, papers)

    tracer.record_tool(
        name="search_arxiv_papers",
        input_payload={"query": user_query, "limit": 5},
        output_payload={"papers": [paper.id for paper in papers]},
    )
    tracer.record_result(answer=answer)
    tracer.close()

    return AgentTurnResult(
        answer=answer,
        citations_text=citations_text,
        citations=citations,
        metadata=(("trace_id", tracer.trace_id or ""),),
    )