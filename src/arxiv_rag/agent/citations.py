from __future__ import annotations

from arxiv_rag.agent.models import RetrievedPaper


def render_citations(papers: list[RetrievedPaper]) -> tuple[str, tuple[RetrievedPaper, ...]]:
    if not papers:
        return "", ()

    citations = []
    for i, paper in enumerate(papers, 1):
        citations.append(f"[{i}] {paper.title} ({paper.id})")

    return "\n".join(citations), tuple(papers)