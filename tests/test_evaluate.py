from __future__ import annotations

import json
import csv

from arxiv_rag.evaluate import (
    EvalQuery,
    _format_report,
    format_query_inspection,
    save_report_artifacts,
    evaluate_queries,
    load_eval_queries,
)


def test_load_eval_queries_reads_expected_shape(tmp_path):
    dataset_path = tmp_path / "eval_queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "query": "category learning",
                    "relevant_ids": ["2403.03835", "1304.3432"],
                }
            ]
        )
    )

    queries = load_eval_queries(dataset_path)

    assert queries == [
        EvalQuery(
            query="category learning",
            relevant_ids=("2403.03835", "1304.3432"),
        )
    ]


def test_evaluate_queries_computes_recall_and_mrr():
    queries = [
        EvalQuery(query="category learning", relevant_ids=("2403.03835",)),
        EvalQuery(query="matrix factorization", relevant_ids=("2012.10853",)),
    ]

    class StubQueryEngine:
        def search(self, query: str, limit: int = 10):
            if query == "category learning":
                return [
                    {"id": "x"},
                    {"id": "2403.03835"},
                ]
            return [
                {"id": "a"},
                {"id": "b"},
            ]

    report = evaluate_queries(StubQueryEngine(), queries, ks=(1, 5))

    assert report["query_count"] == 2
    assert report["recall_at"][1] == 0.0
    assert report["recall_at"][5] == 0.5
    assert report["mrr"] == 0.25
    assert report["per_query"][0]["first_relevant_rank"] == 2
    assert report["per_query"][1]["first_relevant_rank"] is None


def test_format_report_adds_plain_english_interpretation():
    report = {
        "query_count": 4,
        "recall_hits": {5: 2, 10: 2},
        "recall_at": {5: 0.5, 10: 0.5},
        "mrr": 0.375,
        "per_query": [
            {
                "query": "best query",
                "relevant_ids": ("a",),
                "result_ids": ("a", "x"),
                "first_relevant_rank": 1,
            },
            {
                "query": "second query",
                "relevant_ids": ("b",),
                "result_ids": ("x", "b"),
                "first_relevant_rank": 2,
            },
            {
                "query": "miss one",
                "relevant_ids": ("c",),
                "result_ids": ("x", "y"),
                "first_relevant_rank": None,
            },
            {
                "query": "miss two",
                "relevant_ids": ("d",),
                "result_ids": ("x", "y"),
                "first_relevant_rank": None,
            },
        ],
    }

    output = _format_report(report)

    assert "Recall@5: 0.500 (2/4 queries)" in output
    assert "Recall@10: 0.500 (2/4 queries)" in output
    assert "MRR: 0.375" in output
    assert "Relevant paper found in top 5 for 2 of 4 queries." in output
    assert "No additional relevant hits appeared between ranks 6 and 10." in output
    assert "Best query: best query (rank 1)" in output
    assert "Worst queries (no relevant result in top 10):" in output
    assert "- miss one" in output
    assert "- miss two" in output
    assert "Recommendation: try a stronger embedding model." in output


def test_format_query_inspection_shows_hits_and_misses():
    output = format_query_inspection(
        query="category learning and human categorization",
        expected_ids=("2403.03835", "1304.3432"),
        results=[
            {"id": "2403.03835", "title": "Cobweb"},
            {"id": "x", "title": "Other paper"},
        ],
    )

    assert "Query: category learning and human categorization" in output
    assert "Expected relevant IDs: 2403.03835, 1304.3432" in output
    assert "1. 2403.03835 | Cobweb" in output
    assert "2. x | Other paper" in output
    assert "- 2403.03835: found at rank 1" in output
    assert "- 1304.3432: not found in top results" in output


def test_format_query_inspection_includes_source_when_present():
    output = format_query_inspection(
        query="category learning and human categorization",
        expected_ids=("2403.03835",),
        results=[
            {
                "id": "2403.03835",
                "title": "Cobweb",
                "source": "both",
            },
            {
                "id": "x",
                "title": "Other paper",
                "source": "keyword",
            },
        ],
    )

    assert "1. 2403.03835 | Cobweb | source=both" in output
    assert "2. x | Other paper | source=keyword" in output


def test_save_report_artifacts_writes_markdown_json_and_csv(tmp_path):
    report = {
        "query_count": 2,
        "recall_hits": {5: 1, 10: 1},
        "recall_at": {5: 0.5, 10: 0.5},
        "mrr": 0.5,
        "per_query": [
            {
                "query": "query one",
                "relevant_ids": ("a",),
                "result_ids": ("a", "x"),
                "first_relevant_rank": 1,
            },
            {
                "query": "query two",
                "relevant_ids": ("b",),
                "result_ids": ("x", "y"),
                "first_relevant_rank": None,
            },
        ],
    }
    metadata = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "dataset_path": "data/eval_queries.json",
        "timestamp_utc": "2026-05-16T12:00:00Z",
    }

    output_dir = save_report_artifacts(tmp_path, report, metadata)

    summary_md = output_dir / "summary.md"
    summary_json = output_dir / "summary.json"
    per_query_csv = output_dir / "per_query.csv"

    assert summary_md.exists()
    assert summary_json.exists()
    assert per_query_csv.exists()
    assert "sentence-transformers/all-MiniLM-L6-v2" in summary_md.read_text()

    summary_payload = json.loads(summary_json.read_text())
    assert summary_payload["metadata"]["embedding_dim"] == 384
    assert summary_payload["report"]["query_count"] == 2

    with per_query_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["query"] == "query one"
    assert rows[0]["first_relevant_rank"] == "1"
    assert rows[1]["first_relevant_rank"] == ""
