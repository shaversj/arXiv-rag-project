import pytest
from arxiv_rag.ingest import PostgresStore


def test_create_schema():
    store = PostgresStore()
    store.init_schema()

    conn = store.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = {row[0] for row in cursor.fetchall()}
    assert "papers" in tables
    assert "paper_embeddings" in tables
    conn.close()
    store.close()


def test_insert_paper():
    store = PostgresStore()
    store.init_schema()

    paper = {
        "id": "0704.0001",
        "title": "Test Paper",
        "authors": "John Doe",
        "abstract": "Test abstract",
        "categories": "cs.AI",
        "submitter": "Test",
        "journal_ref": None,
        "doi": None,
        "update_date": "2007-04-02"
    }
    embedding = [0.1] * 384
    store.insert_paper(paper, embedding)

    result = store.get_paper("0704.0001")
    assert result is not None
    assert result["id"] == "0704.0001"
    store.close()


def test_search_by_keyword():
    store = PostgresStore()
    store.init_schema()

    paper = {
        "id": "0704.0002",
        "title": "Machine Learning Methods UniqueKeywordAlpha",
        "authors": "Jane Smith",
        "abstract": "This paper discusses neural networks and deep learning approaches with UniqueKeywordAlpha",
        "categories": "cs.AI",
        "submitter": "Test",
        "journal_ref": None,
        "doi": None,
        "update_date": "2007-04-02"
    }
    embedding = [0.1] * 384
    store.insert_paper(paper, embedding)

    results = store.search_by_keyword("UniqueKeywordAlpha")
    assert len(results) >= 1
    assert any(r["id"] == "0704.0002" for r in results)
    store.close()


def test_init_schema_uses_configured_embedding_dimension(monkeypatch):
    executed_sql: list[str] = []

    class FakeCursor:
        def execute(self, sql, params=None):
            executed_sql.append(sql)

        def close(self):
            return None

    class FakeConnection:
        closed = False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            return None

    monkeypatch.setattr(PostgresStore, "_connect", lambda self: setattr(self, "conn", FakeConnection()))

    store = PostgresStore(embedding_dim=1024)
    store.init_schema()

    assert any("embedding vector(1024)" in sql for sql in executed_sql)
