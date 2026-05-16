import pytest
from query_engine import QueryEngine

def test_search_requires_init():
    qe = QueryEngine()
    with pytest.raises(RuntimeError):
        qe.search("test query")

def test_search_integration():
    # This test requires actual data, mark as integration
    pass


def test_initialize_passes_embedding_dimension(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'db_host: "localhost"',
                "db_port: 5432",
                'db_name: "arxiv_rag"',
                'db_user: "postgres"',
                'db_password: "postgres"',
                'embedding_model: "sentence-transformers/all-MiniLM-L6-v2"',
                "embedding_dim: 1024",
                "top_k: 10",
            ]
        )
    )

    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def init_schema(self):
            captured["init_schema_called"] = True

    monkeypatch.setattr("arxiv_rag.ingest.PostgresStore", FakeStore)
    monkeypatch.setattr(
        "query_engine.SentenceTransformer",
        lambda model_name: model_name,
    )

    qe = QueryEngine(config_path=str(config_path))
    qe.initialize()

    assert captured["embedding_dim"] == 1024
    assert captured["init_schema_called"] is True
