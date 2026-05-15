from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedPaper:
    id: str
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = field(default_factory=tuple)
    categories: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AgentTurnResult:
    answer: str
    citations_text: str
    citations: tuple[RetrievedPaper, ...]
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)