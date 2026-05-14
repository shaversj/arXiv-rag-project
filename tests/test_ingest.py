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
        "title": "Machine Learning Methods",
        "authors": "Jane Smith",
        "abstract": "This paper discusses neural networks and deep learning approaches",
        "categories": "cs.AI",
        "submitter": "Test",
        "journal_ref": None,
        "doi": None,
        "update_date": "2007-04-02"
    }
    embedding = [0.1] * 384
    store.insert_paper(paper, embedding)

    results = store.search_by_keyword("machine learning")
    assert len(results) >= 1
    assert any(r["id"] == "0704.0002" for r in results)
    store.close()