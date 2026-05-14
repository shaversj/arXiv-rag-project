from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_search_endpoint():
    response = client.post("/api/search", json={"query": "machine learning"})
    assert response.status_code == 200
    assert "results" in response.json()