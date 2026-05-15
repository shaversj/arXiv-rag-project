"""CLI for arXiv RAG agent."""

import asyncio
import sys
from pathlib import Path

# Load .env before importing Langfuse
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from arxiv_rag.agent.service import run_agent_turn
from arxiv_rag.agent.tools import RetrievalTool
from arxiv_rag.query_engine import QueryEngine


async def main(query: str) -> None:
    query_engine = QueryEngine()
    query_engine.initialize()
    retrieval = RetrievalTool(query_engine)

    result = await run_agent_turn(
        [{"role": "user", "content": query}],
        retrieval_tool=retrieval,
    )

    print("\n--- Answer ---")
    print(result.answer)
    print("\n--- Citations ---")
    print(result.citations_text)
    if result.metadata:
        trace_id = dict(result.metadata).get("trace_id", "")
        if trace_id:
            print(f"\n--- Langfuse Trace ---")
            print(f"https://cloud.langfuse.com/project/.../traces/{trace_id}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m arxiv_rag.agent.cli "What are retrieval agents?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(main(query))