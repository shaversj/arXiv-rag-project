from pathlib import Path

import pytest
import numpy as np
import yaml

import query_engine
from query_engine import QueryEngine


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


def test_search_requires_init():
    qe = QueryEngine()
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
    qe = _make_initialized_engine(tmp_path, monkeypatch, retrieval_mode="semantic")
    semantic_calls: list[tuple[str, int]] = []

    class FakeCursor:
        def execute(self, *args, **kwargs):
            return None

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
    qe = _make_initialized_engine(tmp_path, monkeypatch, retrieval_mode="keyword")
    keyword_calls: list[tuple[str, int]] = []

    class FakeCursor:
        def execute(self, *args, **kwargs):
            return None

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
    qe = _make_initialized_engine(
        tmp_path,
        monkeypatch,
        retrieval_mode="hybrid",
        hybrid_candidate_pool=7,
        hybrid_semantic_weight=0.6,
        hybrid_keyword_weight=0.4,
    )
    calls: list[tuple[str, str, int]] = []
    normalize_calls: list[list[dict[str, object]]] = []
    fuse_calls: list[tuple[list[dict[str, object]], list[dict[str, object]], float, float]] = []

    class FakeCursor:
        def execute(self, *args, **kwargs):
            return None

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
        lambda rows: normalize_calls.append(rows) or rows,
        raising=False,
    )
    monkeypatch.setattr(
        qe,
        "_fuse_results",
        lambda semantic_rows, keyword_rows, semantic_weight, keyword_weight: fuse_calls.append(
            (semantic_rows, keyword_rows, semantic_weight, keyword_weight)
        )
        or [{"id": "a", "title": "Semantic", "score": semantic_weight, "source": "semantic"}],
        raising=False,
    )

    results = qe.search("hybrid query", limit=2)

    assert results == [{"id": "a", "title": "Semantic", "score": 0.6, "source": "semantic"}]
    assert calls == [("semantic", "hybrid query", 7), ("keyword", "hybrid query", 7)]
    assert len(normalize_calls) == 2
    assert len(fuse_calls) == 1
    assert fuse_calls[0][2:] == (0.6, 0.4)


def test_hybrid_deduplicates_by_paper_id(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "42",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 1.0,
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
            "score": 0.4,
            "source": "keyword",
        }
    ]

    fused = qe._fuse_results(semantic_rows, keyword_rows, 0.7, 0.3)

    assert [row["id"] for row in fused] == ["42"]
    assert fused[0]["source"] == "both"


def test_shared_paper_gets_fused_preference(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "shared",
            "title": "Shared",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 1.0,
            "source": "semantic",
        },
        {
            "id": "semantic_only",
            "title": "Semantic Only",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.6,
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
            "score": 1.0,
            "source": "keyword",
        }
    ]

    fused = qe._fuse_results(semantic_rows, keyword_rows, 0.7, 0.3)

    assert fused[0]["id"] == "shared"
    assert fused[0]["score"] == pytest.approx(1.0)


def test_weight_driven_ranking_changes_order(tmp_path, monkeypatch):
    qe = _make_initialized_engine(tmp_path, monkeypatch)
    semantic_rows = [
        {
            "id": "semantic_favorite",
            "title": "Semantic Favorite",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 1.0,
            "source": "semantic",
        },
        {
            "id": "keyword_favorite",
            "title": "Keyword Favorite",
            "authors": "B",
            "abstract": "K",
            "categories": "cs.LG",
            "score": 0.2,
            "source": "semantic",
        },
    ]
    keyword_rows = [
        {
            "id": "semantic_favorite",
            "title": "Semantic Favorite",
            "authors": "A",
            "abstract": "S",
            "categories": "cs.AI",
            "score": 0.1,
            "source": "keyword",
        },
        {
            "id": "keyword_favorite",
            "title": "Keyword Favorite",
            "authors": "B",
            "abstract": "K",
            "categories": "cs.LG",
            "score": 1.0,
            "source": "keyword",
        },
    ]

    semantic_heavy = qe._fuse_results(semantic_rows, keyword_rows, 0.8, 0.2)
    keyword_heavy = qe._fuse_results(semantic_rows, keyword_rows, 0.2, 0.8)

    assert semantic_heavy[0]["id"] == "semantic_favorite"
    assert keyword_heavy[0]["id"] == "keyword_favorite"
