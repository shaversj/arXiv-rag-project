# arXiv RAG Pipeline Implementation Plan (PostgreSQL + pgvector)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG pipeline that ingests arXiv cs.AI metadata (~176k papers) and provides a chat interface for semantic search.

**Architecture:** Python scripts + PostgreSQL (metadata + embeddings + full-text search) + FastAPI + vanilla HTML/JS chat UI. Ingest reads JSON, stores metadata with embeddings in PostgreSQL using pgvector for efficient similarity search. Query uses pgvector's `<->` (L2 distance) operator for semantic search combined with SQL filters.

**Tech Stack:** Python 3.11+, uv (package manager), Docker, PostgreSQL 15+ with pgvector extension, sentence-transformers, FastAPI, vanilla HTML/JS

---

## File Structure

```
/Users/wu36/Code/arXiv-rag-project/
├── config.yaml              # Configuration (database connection)
├── pyproject.toml           # Python project with uv
├── Dockerfile               # Docker container (app only)
├── docker-compose.yaml      # Docker Compose (app + postgres)
├── src/arxiv_rag/
│   ├── ingest.py             # JSON → PostgreSQL (metadata + embeddings)
│   └── __init__.py
├── query_engine.py           # Search logic (pgvector + SQL)
├── app.py                   # FastAPI server
├── static/
│   └── index.html            # Chat UI
└── tests/
    ├── __init__.py
    ├── test_ingest.py
    ├── test_query_engine.py
    └── test_app.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `Dockerfile`
- Create: `docker-compose.yaml`
- Create: `tests/`
- Create: `static/`

- [ ] **Step 1: Create pyproject.toml with uv**

```toml
[project]
name = "arxiv-rag"
version = "0.1.0"
description = "arXiv RAG pipeline with chat interface"
requires-python = ">=3.11"
dependencies = [
    "psycopg2-binary>=2.9.0",
    "sqlalchemy>=2.0.0",
    "sentence-transformers>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "pyyaml>=6.0",
    "tqdm>=4.60",
    "numpy>=1.21",
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/arxiv_rag"]
```

- [ ] **Step 2: Create config.yaml**

```yaml
json_file: "arxiv-metadata-oai-snapshot.json"
category_filter: "cs.AI"
db_host: "localhost"
db_port: 5432
db_name: "arxiv_rag"
db_user: "postgres"
db_password: "postgres"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dim: 384
batch_size: 1000
top_k: 10
```

- [ ] **Step 3: Create Dockerfile (app only, no postgres)**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Create docker-compose.yaml with postgres**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: arxiv_rag
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./arxiv-metadata-oai-snapshot.json:/app/arxiv-metadata-oai-snapshot.json:ro
    environment:
      - PYTHONPATH=/app
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=arxiv_rag
      - DB_USER=postgres
      - DB_PASSWORD=postgres
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

- [ ] **Step 5: Create directories**

```bash
mkdir -p static tests
touch tests/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config.yaml Dockerfile docker-compose.yaml tests/ static/
git commit -m "feat: project setup with PostgreSQL + pgvector"
```

---

## Task 2: Ingest Script

**Files:**
- Create: `src/arxiv_rag/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write test for PostgreSQL storage**

```python
import pytest
from arxiv_rag.ingest import PostgresStore

def test_create_schema():
    store = PostgresStore()
    store.init_schema()
    
    conn = store.get_connection()
    cursor = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = {row[0] for row in cursor.fetchall()}
    assert "papers" in tables
    assert "paper_embeddings" in tables
    conn.close()

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py::test_create_schema -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write PostgresStore class**

```python
import os
import json
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

class PostgresStore:
    def __init__(self, host=None, port=5432, dbname="arxiv_rag",
                 user="postgres", password="postgres"):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", 5432))
        self.dbname = dbname or os.getenv("DB_NAME", "arxiv_rag")
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "postgres")
        self.conn = None
        self._connect()

    def _connect(self):
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )

    def get_connection(self):
        if not self.conn or self.conn.closed:
            self._connect()
        return self.conn

    def init_schema(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                categories TEXT,
                submitter TEXT,
                journal_ref TEXT,
                doi TEXT,
                update_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create paper_embeddings table with vector column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_embeddings (
                paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                embedding vector(384),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on embedding for similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON paper_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        # Create full-text search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_fts 
            ON papers USING gin(to_tsvector('english', title || ' ' || authors || ' ' || COALESCE(abstract, '')))
        """)
        
        conn.commit()
        cursor.close()

    def insert_paper(self, paper, embedding):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Insert paper metadata
        cursor.execute("""
            INSERT INTO papers (id, title, authors, abstract, categories, submitter, journal_ref, doi, update_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                abstract = EXCLUDED.abstract,
                categories = EXCLUDED.categories
        """, (
            paper["id"],
            paper["title"],
            paper["authors"],
            paper["abstract"],
            paper["categories"],
            paper.get("submitter"),
            paper.get("journal_ref"),
            paper.get("doi"),
            paper.get("update_date")
        ))
        
        # Insert embedding
        embedding_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
        cursor.execute("""
            INSERT INTO paper_embeddings (paper_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (paper_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
        """, (paper["id"], embedding_json))
        
        conn.commit()
        cursor.close()

    def get_paper(self, paper_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "authors": row[2],
            "abstract": row[3],
            "categories": row[4],
            "submitter": row[5],
            "journal_ref": row[6],
            "doi": row[7],
            "update_date": row[8]
        }

    def search_by_keyword(self, query, limit=10):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, authors, abstract, categories
            FROM papers
            WHERE to_tsvector('english', title || ' ' || authors || ' ' || COALESCE(abstract, '')) @@ plainto_tsquery('english', %s)
            LIMIT %s
        """, (query, limit))
        results = [
            {"id": row[0], "title": row[1], "authors": row[2], "abstract": row[3], "categories": row[4]}
            for row in cursor.fetchall()
        ]
        cursor.close()
        return results

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py::test_create_schema tests/test_ingest.py::test_insert_paper -v`
Expected: PASS (requires PostgreSQL running)

- [ ] **Step 5: Write main ingest function**

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def ingest(config_path="config.yaml"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    store = PostgresStore(
        host=config.get("db_host"),
        port=config.get("db_port"),
        dbname=config.get("db_name"),
        user=config.get("db_user"),
        password=config.get("db_password")
    )
    store.init_schema()
    
    model = SentenceTransformer(config["embedding_model"])

    total = 0
    ingested = 0
    errors = 0
    batch_papers = []
    batch_embeddings = []

    with open(config["json_file"]) as f:
        for line in tqdm(f, desc="Ingesting"):
            total += 1
            try:
                doc = json.loads(line)
                if config["category_filter"] not in doc.get("categories", "").split():
                    continue

                paper = {
                    "id": doc["id"],
                    "title": doc["title"],
                    "authors": doc["authors"],
                    "abstract": doc["abstract"],
                    "categories": doc["categories"],
                    "submitter": doc.get("submitter"),
                    "journal_ref": doc.get("journal-ref"),
                    "doi": doc.get("doi"),
                    "update_date": doc.get("update_date")
                }

                text = f"{doc['title']} {doc['abstract']}"
                embedding = model.encode(text).astype('float32')

                batch_papers.append(paper)
                batch_embeddings.append(embedding)
                ingested += 1

                if len(batch_papers) >= config["batch_size"]:
                    for p, e in zip(batch_papers, batch_embeddings):
                        store.insert_paper(p, e)
                    batch_papers = []
                    batch_embeddings = []
                    
                    if ingested % 5000 == 0:
                        print(f"Ingested {ingested} papers...")

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error: {e}")

        # Flush remaining
        for p, e in zip(batch_papers, batch_embeddings):
            store.insert_paper(p, e)

    print(f"Done. Total: {total}, Ingested: {ingested}, Errors: {errors}")
    store.close()


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 6: Commit**

```bash
git add src/arxiv_rag/ingest.py tests/test_ingest.py
git commit -m "feat: add ingest script with PostgreSQL + pgvector"
```

---

## Task 3: Query Engine

**Files:**
- Create: `query_engine.py`
- Create: `tests/test_query_engine.py`

- [ ] **Step 1: Write test for search**

```python
import pytest
from query_engine import QueryEngine

def test_search_requires_init():
    qe = QueryEngine()
    with pytest.raises(RuntimeError):
        qe.search("test query")

def test_search_returns_results():
    qe = QueryEngine()
    qe.initialize()
    results = qe.search("machine learning", limit=5)
    assert isinstance(results, list)
    qe.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_query_engine.py -v`
Expected: FAIL - module not found

- [ ] **Step 3: Write QueryEngine class**

```python
import yaml
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class QueryEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.store = None
        self.model = None

    def initialize(self):
        from arxiv_rag.ingest import PostgresStore
        
        self.store = PostgresStore(
            host=self.config.get("db_host"),
            port=self.config.get("db_port"),
            dbname=self.config.get("db_name"),
            user=self.config.get("db_user"),
            password=self.config.get("db_password")
        )
        self.store.init_schema()
        self.model = SentenceTransformer(self.config["embedding_model"])

    def search(self, query, filters=None, limit=None):
        if not self.store:
            self.initialize()

        if limit is None:
            limit = self.config["top_k"]

        # Generate query embedding
        query_embedding = self.model.encode(query).astype('float32')
        
        conn = self.store.get_connection()
        cursor = conn.cursor()
        
        # Search using pgvector's <=> operator for cosine distance
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_list = query_norm.tolist()
        
        cursor.execute("""
            SELECT p.id, p.title, p.authors, p.abstract, p.categories,
                   1 - (e.embedding <=> %s::vector) as score
            FROM papers p
            JOIN paper_embeddings e ON p.id = e.paper_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """, (json.dumps(query_list), json.dumps(query_list), limit * 2))
        
        results = []
        seen_ids = set()
        for row in cursor.fetchall():
            paper_id = row[0]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            results.append({
                "id": row[0],
                "title": row[1],
                "authors": row[2],
                "abstract": row[3],
                "categories": row[4],
                "score": float(row[5])
            })
        
        cursor.close()
        return results[:limit]

    def close(self):
        if self.store:
            self.store.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_query_engine.py -v`
Expected: PASS (mocked or skip integration tests)

- [ ] **Step 5: Commit**

```bash
git add query_engine.py tests/test_query_engine.py
git commit -m "feat: add query engine with PostgreSQL pgvector"
```

---

## Task 4: FastAPI Server

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write test for API endpoints**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL - app not found

- [ ] **Step 3: Write FastAPI app**

```python
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from query_engine import QueryEngine

app = FastAPI()
qe = QueryEngine()

class SearchRequest(BaseModel):
    query: str
    filters: dict = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/search")
def search(req: SearchRequest):
    try:
        results = qe.search(req.query, req.filters)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/paper/{paper_id}")
def get_paper(paper_id: str):
    qe.initialize()
    paper = qe.store.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.on_event("shutdown")
def shutdown_event():
    qe.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add FastAPI server with search endpoints"
```

---

## Task 5: Chat UI

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: Write chat interface HTML** (same as before)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv RAG Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
        #header { background: #16213e; padding: 1rem; border-bottom: 1px solid #0f3460; }
        #header h1 { font-size: 1.2rem; }
        #chat { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
        .message { max-width: 80%; padding: 0.75rem 1rem; border-radius: 8px; }
        .user { align-self: flex-end; background: #0f3460; }
        .assistant { align-self: flex-start; background: #16213e; }
        .paper-card { background: #0f3460; padding: 1rem; border-radius: 8px; margin-top: 0.5rem; }
        .paper-card h3 { font-size: 1rem; margin-bottom: 0.5rem; }
        .paper-card .meta { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; }
        .paper-card .abstract { font-size: 0.9rem; line-height: 1.4; }
        .paper-card a { color: #60a5fa; text-decoration: none; }
        .paper-card a:hover { text-decoration: underline; }
        #input-area { background: #16213e; padding: 1rem; border-top: 1px solid #0f3460; display: flex; gap: 0.5rem; }
        #query-input { flex: 1; padding: 0.75rem; border-radius: 8px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; font-size: 1rem; }
        #query-input:focus { outline: none; border-color: #60a5fa; }
        #send-btn { padding: 0.75rem 1.5rem; background: #60a5fa; color: #1a1a2e; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
        #send-btn:hover { background: #3b82f6; }
        #send-btn:disabled { background: #4a5568; cursor: not-allowed; }
        .loading { color: #94a3b8; font-style: italic; }
    </style>
</head>
<body>
    <div id="header"><h1>arXiv RAG Chat</h1></div>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="query-input" placeholder="Ask about papers..." />
        <button id="send-btn">Send</button>
    </div>
    <script>
        const chat = document.getElementById("chat");
        const queryInput = document.getElementById("query-input");
        const sendBtn = document.getElementById("send-btn");

        async function sendMessage() {
            const query = queryInput.value.trim();
            if (!query) return;
            addMessage("user", query);
            queryInput.value = "";
            sendBtn.disabled = true;
            addMessage("assistant", "Searching...", true);
            try {
                const response = await fetch("/api/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query })
                });
                const data = await response.json();
                chat.removeChild(chat.lastChild);
                if (data.results && data.results.length > 0) {
                    addMessage("assistant", `Found ${data.results.length} papers:`);
                    data.results.forEach(paper => addPaperCard(paper));
                } else {
                    addMessage("assistant", "No papers found matching your query.");
                }
            } catch (err) {
                chat.removeChild(chat.lastChild);
                addMessage("assistant", "Error: " + err.message);
            }
            sendBtn.disabled = false;
        }

        function addMessage(role, content, isLoading = false) {
            const div = document.createElement("div");
            div.className = `message ${role}`;
            div.textContent = content;
            if (isLoading) div.classList.add("loading");
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addPaperCard(paper) {
            const div = document.createElement("div");
            div.className = "paper-card";
            div.innerHTML = `
                <h3><a href="https://arxiv.org/abs/${paper.id}" target="_blank">${paper.title}</a></h3>
                <div class="meta">${paper.authors} | Score: ${(paper.score * 100).toFixed(1)}%</div>
                <div class="abstract">${paper.abstract ? paper.abstract.substring(0, 300) + "..." : "No abstract"}</div>
            `;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        sendBtn.addEventListener("click", sendMessage);
        queryInput.addEventListener("keypress", e => { if (e.key === "Enter") sendMessage(); });
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify file exists**

Run: `ls -la static/index.html`

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: add chat UI"
```

---

## Task 6: Integration Test

**Files:**
- None (run existing tests + manual verify)

- [ ] **Step 1: Start PostgreSQL with docker-compose**

```bash
docker-compose up -d postgres
```

- [ ] **Step 2: Wait for postgres to be healthy, then run tests**

```bash
docker-compose up -d
docker exec arxiv-rag-postgres-1 pg_isready -U postgres
docker-compose run --rm app pytest tests/ -v
```

- [ ] **Step 3: Run ingest**

```bash
docker-compose run --rm app uv run python3 -m arxiv_rag.ingest
```

- [ ] **Step 4: Open browser**

Open http://localhost:8000 and test a query.

---

## Dependencies

```toml
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
sentence-transformers>=2.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
pyyaml>=6.0
tqdm>=4.60
numpy>=1.21
pytest>=7.0
pytest-asyncio>=0.21.0
```

---

## Execution Order

1. Task 1: Project Setup (pyproject.toml, config.yaml, Dockerfile, docker-compose.yaml)
2. Task 2: Ingest Script (PostgresStore + ingest function)
3. Task 3: Query Engine (pgvector similarity search)
4. Task 4: FastAPI Server
5. Task 5: Chat UI
6. Task 6: Integration Test

---

## Key PostgreSQL + pgvector Benefits

- **Single database** for metadata + embeddings + full-text search
- **IVF index** on embeddings for fast similarity search at scale
- **ACID compliance** for data integrity
- **Concurrent access** from multiple clients
- **Easy backup/restore** with standard PostgreSQL tools