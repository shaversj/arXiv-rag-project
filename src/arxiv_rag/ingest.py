import os
import json
import psycopg2


class PostgresStore:
    def __init__(self, host=None, port=5432, dbname="arxiv_rag",
                 user="postgres", password="postgres", embedding_dim=384):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", 5432))
        self.dbname = dbname or os.getenv("DB_NAME", "arxiv_rag")
        self.user = user or os.getenv("DB_USER", "postgres")
        self.password = password or os.getenv("DB_PASSWORD", "postgres")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", 384))
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
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS paper_embeddings (
                paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                embedding vector({self.embedding_dim}),
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

        # Insert embedding as JSON string that PostgreSQL casts to vector
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        cursor.execute("""
            INSERT INTO paper_embeddings (paper_id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (paper_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
        """, (paper["id"], json.dumps(embedding_list)))

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


def ingest(config_path="config.yaml"):
    import yaml
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    with open(config_path) as f:
        config = yaml.safe_load(f)

    store = PostgresStore(
        host=config.get("db_host"),
        port=config.get("db_port"),
        dbname=config.get("db_name"),
        user=config.get("db_user"),
        password=config.get("db_password"),
        embedding_dim=config.get("embedding_dim", 384),
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
