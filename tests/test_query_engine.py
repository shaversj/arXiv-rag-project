from pathlib import Path

import pytest
import numpy as np
import yaml

from arxiv_rag import query_engine
from arxiv_rag.query_engine import QueryEngine


def _write_config(tmp_path: Path, **overrides) -> Path:
    config = {
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "arxiv_rag",
        "db_user": "postgres",
        "db_password": "postgres",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "top_k": 10,
        "retrieval_mode": "semantic",
        "hybrid_semantic_weight": 0.7,
        "hybrid_keyword_weight": 0.3,
        "hybrid_candidate_pool": 25,
    }
    config.update(overrides)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _make_initialized_engine(tmp_path: Path, monkeypatch, **config_overrides) -> QueryEngine:
    config_path = _write_config(tmp_path, **config_overrides)
    monkeypatch.setattr(query_engine, "SentenceTransformer", lambda model_name: model_name)

    qe = QueryEngine(config_path=str(config_path))
    qe.store = object()
    qe.model = object()
    return qe


def _make_blocking_search_engine(tmp_path: Path, monkeypatch, **config_overrides) -> QueryEngine:
    qe = _make_initialized_engine(tmp_path, monkeypatch, **config_overrides)

    class FakeCursor:
        def execute(self, *args, **kwargs):
            pytest.fail("legacy SQL path should not run")

        def fetchall(self):
            return []

        def close(self):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        def get_connection(self):
            return FakeConnection()

    class FakeModel:
        def encode(self, query):
            return np.array([1.0, 0.0], dtype=np.float32)

    qe.store = FakeStore()
    qe.model = FakeModel()
    return qe


def test_search_requires_init(tmp_path):
    qe = QueryEngine(config_path=str(_write_config(tmp_path)))
    with pytest.raises(RuntimeError):
        qe.search("test query")


def test_initialize_passes_embedding_dimension(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, embedding_dim=1024)
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def init_schema(self):
            captured["init_schema_called"] = True

    monkeypatch.setattr("arxiv_rag.ingest.PostgresStore", FakeStore)
    monkeypatch.setattr(query_engine, "SentenceTransformer", lambda model_name: model_name)

    qe = QueryEngine(config_path=str(config_path))
    qe.initialize()

    assert captured["embedding_dim"] == 1024
    assert captured["init_schema_called"] is True


def test_search_routes_semantic_mode(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(tmp_path, monkeypatch, retrieval_mode="semantic")
    semantic_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        qe,
        "_search_semantic",
        lambda query, limit: semantic_calls.append((query, limit))
        or [{"id": "s1", "title": "Semantic"}],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_search_keyword",
        lambda query, limit: pytest.fail("keyword branch should not run"),
        raising=False,
    )

    results = qe.search("graph retrieval", limit=3)

    assert results == [{"id": "s1", "title": "Semantic"}]
    assert semantic_calls == [("graph retrieval", 3)]


def test_search_routes_keyword_mode(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(tmp_path, monkeypatch, retrieval_mode="keyword")
    keyword_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        qe,
        "_search_semantic",
        lambda query, limit: pytest.fail("semantic branch should not run"),
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_search_keyword",
        lambda query, limit: keyword_calls.append((query, limit))
        or [{"id": "k1", "title": "Keyword"}],
        raising=False,
    )

    results = qe.search("symbolic regression", limit=4)

    assert results == [{"id": "k1", "title": "Keyword"}]
    assert keyword_calls == [("symbolic regression", 4)]


def test_search_routes_hybrid_mode(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(
        tmp_path,
        monkeypatch,
        retrieval_mode="hybrid",
        hybrid_candidate_pool=7,
        hybrid_semantic_weight=0.6,
        hybrid_keyword_weight=0.4,
    )
    calls: list[tuple[str, str, int]] = []
    fuse_calls: list[
        tuple[list[dict[str, object]], list[dict[str, object]], float, float, int]
    ] = []

    monkeypatch.setattr(
        qe,
        "_search_semantic",
        lambda query, limit: calls.append(("semantic", query, limit))
        or [{"id": "a", "title": "Semantic", "score": 9.0, "source": "semantic"}],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_search_keyword",
        lambda query, limit: calls.append(("keyword", query, limit))
        or [{"id": "b", "title": "Keyword", "score": 3.0, "source": "keyword"}],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_normalize_scores",
        lambda rows: pytest.fail("_normalize_scores should not run in rank-fusion hybrid mode"),
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_fuse_results",
        lambda semantic_rows, keyword_rows, semantic_weight, keyword_weight: pytest.fail(
            "_fuse_results should not run in rank-fusion hybrid mode"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_fuse_ranked_results",
        lambda semantic_rows, keyword_rows, semantic_weight, keyword_weight, rank_constant: fuse_calls.append(
            (semantic_rows, keyword_rows, semantic_weight, keyword_weight, rank_constant)
        )
        or [{"id": "a", "title": "Semantic", "score": semantic_weight, "source": "semantic"}],
        raising=False,
    )

    results = qe.search("hybrid query", limit=2)

    assert results == [{"id": "a", "title": "Semantic", "score": 0.6, "source": "semantic"}]
    assert calls == [("semantic", "hybrid query", 7), ("keyword", "hybrid query", 7)]
    assert len(fuse_calls) == 1
    assert fuse_calls[0][2:] == (0.6, 0.4, 1)


def test_search_routes_hybrid_mode_with_custom_rank_constant(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(
        tmp_path,
        monkeypatch,
        retrieval_mode="hybrid",
        hybrid_candidate_pool=5,
        hybrid_rank_constant=4,
    )
    fuse_calls: list[
        tuple[list[dict[str, object]], list[dict[str, object]], float, float, int]
    ] = []

    monkeypatch.setattr(
        qe,
        "_search_semantic",
        lambda query, limit: [{"id": "a", "title": "Semantic", "score": 9.0, "source": "semantic"}],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_search_keyword",
        lambda query, limit: [{"id": "b", "title": "Keyword", "score": 3.0, "source": "keyword"}],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_fuse_ranked_results",
        lambda semantic_rows, keyword_rows, semantic_weight, keyword_weight, rank_constant: fuse_calls.append(
            (semantic_rows, keyword_rows, semantic_weight, keyword_weight, rank_constant)
        )
        or [{"id": "a", "title": "Semantic", "score": 0.0, "source": "semantic"}],
        raising=False,
    )

    qe.search("hybrid query", limit=2)

    assert fuse_calls[0][2:] == (0.7, 0.3, 4)


def test_search_hybrid_uses_limit_when_above_candidate_pool(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(
        tmp_path,
        monkeypatch,
        retrieval_mode="hybrid",
        hybrid_candidate_pool=3,
    )
    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        qe,
        "_search_semantic",
        lambda query, limit: calls.append(("semantic", query, limit)) or [],
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_search_keyword",
        lambda query, limit: calls.append(("keyword", query, limit)) or [],
        raising=False,
    )

    qe.search("wide hybrid query", limit=8)

    assert calls == [("semantic", "wide hybrid query", 8), ("keyword", "wide hybrid query", 8)]


def test_search_rejects_unsupported_retrieval_mode(monkeypatch, tmp_path):
    qe = _make_blocking_search_engine(tmp_path, monkeypatch, retrieval_mode="mystery")

    with pytest.raises(ValueError, match="Unsupported retrieval_mode: mystery"):
        qe.search("invalid mode")


def test_rank_fusion_combines_duplicate_paper_by_id(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "42",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.9,
            "source": "semantic",
        }
    ]
    keyword_rows = [
        {
            "id": "42",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.1,
            "source": "keyword",
        }
    ]

    fused = qe._fuse_ranked_results(semantic_rows, keyword_rows, 0.7, 0.3, 1)

    assert [row["id"] for row in fused] == ["42"]
    assert fused[0]["source"] == "both"
    assert fused[0]["score"] == pytest.approx(0.5)


def test_rank_fusion_prefers_overlap_even_if_raw_scores_differ(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "semantic_only",
            "title": "Semantic Only",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.99,
            "source": "semantic",
        },
        {
            "id": "shared",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.01,
            "source": "semantic",
        },
    ]
    keyword_rows = [
        {
            "id": "shared",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.02,
            "source": "keyword",
        }
    ]

    fused = qe._fuse_ranked_results(semantic_rows, keyword_rows, 0.7, 0.3, 1)

    assert fused[0]["id"] == "shared"
    assert fused[0]["score"] == pytest.approx((0.7 / 3) + (0.3 / 2))


def test_rank_fusion_ignores_raw_score_magnitude_and_uses_rank_order(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "semantic_top",
            "title": "Semantic Top",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.0001,
            "source": "semantic",
        },
        {
            "id": "keyword_top",
            "title": "Keyword Top",
            "authors": "B",
            "abstract": "K",
            "categories": "cs.LG",
            "score": 999.0,
            "source": "semantic",
        },
    ]
    keyword_rows = [
        {
            "id": "keyword_top",
            "title": "Keyword Top",
            "authors": "B",
            "abstract": "K",
            "categories": "cs.LG",
            "score": 0.0001,
            "source": "keyword",
        },
        {
            "id": "semantic_top",
            "title": "Semantic Top",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 999.0,
            "source": "keyword",
        },
    ]

    fused = qe._fuse_ranked_results(semantic_rows, keyword_rows, 0.3, 0.7, 1)

    assert [row["id"] for row in fused] == ["keyword_top", "semantic_top"]


def test_rank_fusion_rejects_non_positive_rank_constant(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="hybrid_rank_constant must be greater than 0"):
        qe._fuse_ranked_results([], [], 0.7, 0.3, 0)


def test_search_semantic_uses_stable_secondary_sort_by_id(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    class FakeCursor:
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

        def close(self):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        def get_connection(self):
            return FakeConnection()

    class FakeModel:
        def encode(self, query):
            return np.array([1.0, 0.0], dtype=np.float32)

    qe.store = FakeStore()
    qe.model = FakeModel()

    qe._search_semantic("stable ordering", 5)

    sql = captured["sql"]
    assert "ORDER BY e.embedding <=> %s::vector, p.id ASC" in sql


def test_search_keyword_uses_full_text_expression_with_authors(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    class FakeCursor:
        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return []

        def close(self):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        def get_connection(self):
            return FakeConnection()

    qe.store = FakeStore()

    qe._search_keyword("author query", 5)

    sql = captured["sql"]
    assert "p.title || ' ' || p.authors || ' ' || COALESCE(p.abstract, '')" in sql
    assert captured["params"] == ("author query", "author query", 5)
