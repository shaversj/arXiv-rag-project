import pytest
from query_engine import QueryEngine

def test_search_requires_init():
    qe = QueryEngine()
    with pytest.raises(RuntimeError):
        qe.search("test query")

def test_search_integration():
    # This test requires actual data, mark as integration
    pass