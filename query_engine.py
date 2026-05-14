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
            raise RuntimeError("QueryEngine not initialized. Call initialize() first.")

        if limit is None:
            limit = self.config["top_k"]

        # Generate query embedding
        query_embedding = self.model.encode(query).astype('float32')

        conn = self.store.get_connection()
        cursor = conn.cursor()

        # Normalize embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_list = query_norm.tolist()

        # Search using pgvector's <=> operator for cosine distance
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