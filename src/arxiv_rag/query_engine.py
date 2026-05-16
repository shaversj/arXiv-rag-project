import json

import numpy as np
import yaml
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

    def analyze(self, operation="count", group_by="author", time_range="all", query=None, limit=10):
        """Analyze papers with aggregations.

        Args:
            operation: "count" (count papers)
            group_by: "author" (group by author), "category" (group by category)
            time_range: "7d", "30d", "90d", "all"
            query: optional text search to filter papers
            limit: max results to return
        """
        if not self.store:
            raise RuntimeError("QueryEngine not initialized. Call initialize() first.")

        conn = self.store.get_connection()
        cursor = conn.cursor()

        # Build time filter
        time_filter = ""
        if time_range != "all":
            days = {"7d": 7, "30d": 30, "90d": 90}.get(time_range, 30)
            time_filter = f" AND created_at > NOW() - INTERVAL '{days} days'"

        # Build optional text search filter
        text_filter = ""
        if query:
            text_filter = f" AND to_tsvector('english', title || ' ' || COALESCE(abstract, '')) @@ plainto_tsquery('english', %s)"
            query_param = query
        else:
            query_param = None

        if group_by == "author":
            # Unnest the comma-separated authors and group
            sql = f"""
                SELECT unnest(string_to_array(authors, ',')) as author,
                       COUNT(*) as paper_count
                FROM papers
                WHERE authors IS NOT NULL AND authors != '' {time_filter} {text_filter}
                GROUP BY author
                ORDER BY paper_count DESC
                LIMIT %s
            """
            if query_param:
                cursor.execute(sql, (query_param, limit))
            else:
                cursor.execute(sql, (limit,))
            results = [
                {"author": row[0].strip(), "paper_count": row[1]}
                for row in cursor.fetchall()
            ]
        elif group_by == "category":
            sql = f"""
                SELECT unnest(string_to_array(categories, ' ')) as category,
                       COUNT(*) as paper_count
                FROM papers
                WHERE categories IS NOT NULL AND categories != '' {time_filter} {text_filter}
                GROUP BY category
                ORDER BY paper_count DESC
                LIMIT %s
            """
            if query_param:
                cursor.execute(sql, (query_param, limit))
            else:
                cursor.execute(sql, (limit,))
            results = [
                {"category": row[0].strip(), "paper_count": row[1]}
                for row in cursor.fetchall()
            ]
        elif group_by == "date":
            sql = f"""
                SELECT DATE(created_at) as date,
                       COUNT(*) as paper_count
                FROM papers
                WHERE 1=1 {time_filter} {text_filter}
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT %s
            """
            if query_param:
                cursor.execute(sql, (query_param, limit))
            else:
                cursor.execute(sql, (limit,))
            results = [
                {"date": str(row[0]), "paper_count": row[1]}
                for row in cursor.fetchall()
            ]
        else:
            cursor.close()
            raise ValueError(f"Unsupported group_by: {group_by}")

        cursor.close()
        return results

    def close(self):
        if self.store:
            self.store.close()
