from arxiv_rag.agent.models import RetrievedPaper
from arxiv_rag.agent.tools import RetrievalTool, normalize_papers_for_tool


def test_normalize_papers_for_tool_keeps_order_and_truncates_abstracts():
    papers = [
        RetrievedPaper(
            id="1",
            title="A",
            abstract="x" * 400,
            authors=("B",),
            categories=("cs.AI",),
        ),
        RetrievedPaper(
            id="2",
            title="C",
            abstract="short abstract",
            authors=("D", "E"),
            categories=("cs.LG", "cs.CL"),
        )
    ]

    payload = normalize_papers_for_tool(papers)

    assert payload == [
        {
            "id": "1",
            "title": "A",
            "authors": "B",
            "abstract": ("x" * 240) + "...",
            "categories": "cs.AI",
        },
        {
            "id": "2",
            "title": "C",
            "authors": "D, E",
            "abstract": "short abstract",
            "categories": "cs.LG, cs.CL",
        },
    ]


class StubQueryEngine:
    def search(self, query, limit=5):
        assert query == "tooling"
        assert limit == 1
        return [
            {
                "id": "42",
                "title": "Tooling Paper",
                "authors": "Ileana Streinu and Louis Theran",
                "abstract": "Interesting paper",
                "categories": "cs.AI cs.LG",
                "score": 0.987,
                "source": "hybrid",
            }
        ]


def test_retrieval_tool_maps_query_engine_rows():
    tool = RetrievalTool(StubQueryEngine())

    results = tool.search("tooling", limit=1)

    assert results == [
        RetrievedPaper(
            id="42",
            title="Tooling Paper",
            abstract="Interesting paper",
            authors=("Ileana Streinu", "Louis Theran"),
            categories=("cs.AI", "cs.LG"),
        )
    ]
    assert isinstance(results[0], RetrievedPaper)
    assert not hasattr(results[0], "score")
    assert not hasattr(results[0], "source")
