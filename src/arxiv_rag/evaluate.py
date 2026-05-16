from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from arxiv_rag.query_engine import QueryEngine


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: tuple[str, ...]


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text())
    return [
        EvalQuery(
            query=item["query"],
            relevant_ids=tuple(item["relevant_ids"]),
        )
        for item in payload
    ]


def load_eval_metadata(config_path: str | Path, dataset_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text())
    return {
        "embedding_model": config["embedding_model"],
        "embedding_dim": config["embedding_dim"],
        "dataset_path": str(dataset_path),
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def _first_relevant_rank(result_ids: list[str], relevant_ids: tuple[str, ...]) -> int | None:
    relevant_set = set(relevant_ids)
    for index, paper_id in enumerate(result_ids, start=1):
        if paper_id in relevant_set:
            return index
    return None


def evaluate_queries(
    query_engine: Any,
    queries: list[EvalQuery],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    recall_hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    max_k = max(ks, default=10)

    for eval_query in queries:
        results = query_engine.search(eval_query.query, limit=max_k)
        result_ids = [str(row["id"]) for row in results]
        first_rank = _first_relevant_rank(result_ids, eval_query.relevant_ids)

        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

        for k in ks:
            top_k_ids = set(result_ids[:k])
            if top_k_ids.intersection(eval_query.relevant_ids):
                recall_hits[k] += 1

        per_query.append(
            {
                "query": eval_query.query,
                "relevant_ids": eval_query.relevant_ids,
                "result_ids": tuple(result_ids),
                "first_relevant_rank": first_rank,
            }
        )

    query_count = len(queries)
    recall_at = {
        k: (recall_hits[k] / query_count if query_count else 0.0)
        for k in ks
    }
    mrr = reciprocal_rank_sum / query_count if query_count else 0.0

    return {
        "query_count": query_count,
        "recall_hits": recall_hits,
        "recall_at": recall_at,
        "mrr": mrr,
        "per_query": per_query,
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = [f"Queries evaluated: {report['query_count']}"]
    query_count = report["query_count"]
    recall_hits = report.get("recall_hits", {})
    recall_at = report["recall_at"]

    for k, value in recall_at.items():
        hits = recall_hits.get(k, round(value * query_count))
        lines.append(f"Recall@{k}: {value:.3f} ({hits}/{query_count} queries)")
    lines.append(f"MRR: {report['mrr']:.3f}")

    lines.append("")
    top5_hits = recall_hits.get(5, round(recall_at.get(5, 0.0) * query_count))
    lines.append(
        f"Relevant paper found in top 5 for {top5_hits} of {query_count} queries."
    )

    if 10 in recall_at and 5 in recall_at and recall_hits.get(10, 0) == top5_hits:
        lines.append("No additional relevant hits appeared between ranks 6 and 10.")

    ranked = [
        item for item in report["per_query"] if item["first_relevant_rank"] is not None
    ]
    misses = [item for item in report["per_query"] if item["first_relevant_rank"] is None]

    if ranked:
        best_query = min(ranked, key=lambda item: item["first_relevant_rank"])
        lines.append(
            f"Best query: {best_query['query']} (rank {best_query['first_relevant_rank']})"
        )

    if misses:
        lines.append("")
        lines.append("Worst queries (no relevant result in top 10):")
        for item in misses:
            lines.append(f"- {item['query']}")

    lines.append("")
    recall_at_10 = recall_at.get(10, 0.0)
    if recall_at_10 >= 0.8 and report["mrr"] >= 0.6:
        recommendation = "Recommendation: keep this embedding model."
    else:
        recommendation = "Recommendation: try a stronger embedding model."
    lines.append(recommendation)

    return "\n".join(lines)


def format_query_inspection(
    *,
    query: str,
    expected_ids: tuple[str, ...],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"Query: {query}",
        f"Expected relevant IDs: {', '.join(expected_ids)}",
        "",
        "Top results:",
    ]
    for index, row in enumerate(results, start=1):
        result_line = f"{index}. {row['id']} | {row['title']}"
        if row.get("source"):
            result_line += f" | source={row['source']}"
        lines.append(result_line)

    result_ids = [str(row["id"]) for row in results]
    lines.append("")
    lines.append("Expected ID check:")
    for paper_id in expected_ids:
        if paper_id in result_ids:
            rank = result_ids.index(paper_id) + 1
            lines.append(f"- {paper_id}: found at rank {rank}")
        else:
            lines.append(f"- {paper_id}: not found in top results")

    return "\n".join(lines)


def _slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def _report_directory_name(metadata: dict[str, Any]) -> str:
    timestamp = metadata["timestamp_utc"].replace(":", "-").replace("T", "_").replace("Z", "")
    model_slug = _slugify_model_name(metadata["embedding_model"])
    return f"{timestamp}_{model_slug}"


def _format_summary_markdown(report: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Benchmark Summary",
        "",
        f"- Embedding model: `{metadata['embedding_model']}`",
        f"- Embedding dimension: `{metadata['embedding_dim']}`",
        f"- Dataset: `{metadata['dataset_path']}`",
        f"- Timestamp (UTC): `{metadata['timestamp_utc']}`",
        "",
        "```text",
        _format_report(report),
        "```",
    ]
    return "\n".join(lines)


def save_report_artifacts(
    output_root: str | Path,
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    output_root_path = Path(output_root)
    output_dir = output_root_path / _report_directory_name(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.md").write_text(_format_summary_markdown(report, metadata))
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "report": report}, indent=2)
    )

    with (output_dir / "per_query.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query", "relevant_ids", "result_ids", "first_relevant_rank"],
        )
        writer.writeheader()
        for item in report["per_query"]:
            writer.writerow(
                {
                    "query": item["query"],
                    "relevant_ids": "|".join(item["relevant_ids"]),
                    "result_ids": "|".join(item["result_ids"]),
                    "first_relevant_rank": item["first_relevant_rank"] or "",
                }
            )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on labeled queries.")
    parser.add_argument("dataset", help="Path to the evaluation JSON file.")
    parser.add_argument(
        "--output-dir",
        help="Optional directory for saving summary.md, summary.json, and per_query.csv.",
    )
    args = parser.parse_args()

    query_engine = QueryEngine()
    query_engine.initialize()
    try:
        queries = load_eval_queries(args.dataset)
        report = evaluate_queries(query_engine, queries)
    finally:
        query_engine.close()

    print(_format_report(report))
    if args.output_dir:
        metadata = load_eval_metadata("config.yaml", args.dataset)
        output_dir = save_report_artifacts(args.output_dir, report, metadata)
        print(f"\nSaved evaluation artifacts to: {output_dir}")


def inspect_query_main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval results for a single labeled query."
    )
    parser.add_argument("query", help="Query text to search for.")
    parser.add_argument(
        "--expected-id",
        action="append",
        default=[],
        dest="expected_ids",
        help="Expected relevant paper ID. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of top results to print.",
    )
    args = parser.parse_args()

    query_engine = QueryEngine()
    query_engine.initialize()
    try:
        results = query_engine.search(args.query, limit=args.limit)
    finally:
        query_engine.close()

    print(
        format_query_inspection(
            query=args.query,
            expected_ids=tuple(args.expected_ids),
            results=results,
        )
    )


if __name__ == "__main__":
    main()
