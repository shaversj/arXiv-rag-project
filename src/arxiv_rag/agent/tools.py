from __future__ import annotations

import re
from typing import TYPE_CHECKING

from arxiv_rag.agent.models import RetrievedPaper

if TYPE_CHECKING:
    from arxiv_rag.query_engine import QueryEngine

_ABSTRACT_SNIPPET_LIMIT = 240


def _display_text(values: tuple[str, ...] | list[str] | str | None) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return ", ".join(part for part in values if part)


def _split_field(value: str | tuple[str, ...] | list[str] | None, pattern: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(part for part in value if part)

    parts = (segment.strip() for segment in re.split(pattern, value))
    return tuple(part for part in parts if part)


def normalize_papers_for_tool(papers: list[RetrievedPaper]) -> list[dict]:
    normalized: list[dict] = []
    for paper in papers:
        abstract = paper.abstract or ""
        if len(abstract) > _ABSTRACT_SNIPPET_LIMIT:
            abstract = abstract[:_ABSTRACT_SNIPPET_LIMIT] + "..."
        normalized.append(
            {
                "id": paper.id,
                "title": paper.title,
                "authors": _display_text(paper.authors),
                "abstract": abstract,
                "categories": _display_text(paper.categories),
            }
        )
    return normalized


class RetrievalTool:
    def __init__(self, query_engine: "QueryEngine"):
        self.query_engine = query_engine
        self._mcp_server = None
        self._turn_papers: list[RetrievedPaper] = []

    @property
    def mcp_server(self):
        if self._mcp_server is None:
            from arxiv_rag.agent.service import create_retrieval_server
            self._mcp_server = create_retrieval_server(self)
        return self._mcp_server

    def search(self, query: str, limit: int = 5) -> list[RetrievedPaper]:
        rows = self.query_engine.search(query, limit=limit)
        papers = []
        for row in rows:
            if "id" not in row or "title" not in row:
                raise ValueError(
                    f"QueryEngine.search returned a row missing required keys 'id' or 'title': {row!r}"
                )
            papers.append(
                RetrievedPaper(
                    id=str(row["id"]),
                    title=row["title"],
                    abstract=row.get("abstract") or "",
                    authors=_split_field(row.get("authors"), r"\s*,\s*|\s*;\s*|\s+and\s+"),
                    categories=_split_field(row.get("categories"), r"[\s,;]+"),
                )
            )
        self._turn_papers.extend(papers)
        return papers

    def reset_turn_tracking(self) -> None:
        self._turn_papers = []

    def get_turn_retrieved_papers(self) -> tuple[RetrievedPaper, ...]:
        return tuple(self._turn_papers)

    def analyze(self, operation="count", group_by="author", time_range="all", query=None, limit=10):
        return self.query_engine.analyze(
            operation=operation,
            group_by=group_by,
            time_range=time_range,
            query=query,
            limit=limit,
        )
