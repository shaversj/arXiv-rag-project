# Hybrid Retrieval Implementation Checklist

## Goal

Add a second retrieval mode beside embeddings so the project supports:
- `semantic` retrieval
- `keyword` retrieval
- `hybrid` retrieval via late fusion

Target outcome:
- improve `Recall@5`
- improve `Recall@10`
- reduce total misses on `data/eval_queries.json`

## Config

- [ ] Add retrieval settings to [config.yaml](/Users/wu36/Code/arXiv-rag-project/config.yaml)

```yaml
retrieval_mode: "semantic"      # semantic | keyword | hybrid
hybrid_semantic_weight: 0.7
hybrid_keyword_weight: 0.3
hybrid_candidate_pool: 25
```

Rules:
- keep `semantic` as the default mode initially
- only tune weights after the first benchmark comparison

## Query Engine Refactor

File: [src/arxiv_rag/query_engine.py](/Users/wu36/Code/arXiv-rag-project/src/arxiv_rag/query_engine.py)

- [ ] Extract the current embedding search logic into `_search_semantic(query, limit)`
- [ ] Add `_search_keyword(query, limit)` using PostgreSQL full-text search
- [ ] Make both branches return the same row shape:
  - `id`
  - `title`
  - `authors`
  - `abstract`
  - `categories`
  - `score`
  - `source`
- [ ] Add `_normalize_scores(rows)`
- [ ] Add `_fuse_results(semantic_rows, keyword_rows, semantic_weight, keyword_weight)`
- [ ] Update `search()` to route based on `retrieval_mode`

Notes:
- keyword search should use `ts_rank_cd(...)` or equivalent ranking
- hybrid fusion should dedupe by paper ID
- missing branch scores should be treated as `0`
- fused score should be a weighted sum of normalized scores

## SQL / Ranking

Keyword branch requirements:
- [ ] Use `to_tsvector(...) @@ plainto_tsquery(...)`
- [ ] Return a keyword relevance score
- [ ] Sort by keyword score descending

Semantic branch requirements:
- [ ] Preserve cosine-similarity behavior from current implementation
- [ ] Preserve result deduplication by paper ID

Hybrid branch requirements:
- [ ] Fetch top `hybrid_candidate_pool` from semantic
- [ ] Fetch top `hybrid_candidate_pool` from keyword
- [ ] Normalize scores independently by branch
- [ ] Fuse scores using config weights
- [ ] Sort by fused score descending
- [ ] Return top `k`

## Tool Compatibility

File: [src/arxiv_rag/agent/tools.py](/Users/wu36/Code/arXiv-rag-project/src/arxiv_rag/agent/tools.py)

- [ ] Confirm `RetrievalTool.search()` still works with the updated `QueryEngine.search()` return shape
- [ ] Keep existing required keys stable: `id`, `title`, `authors`, `abstract`, `categories`
- [ ] Decide whether to expose `source` later for debugging only

## Evaluation

File: [src/arxiv_rag/evaluate.py](/Users/wu36/Code/arXiv-rag-project/src/arxiv_rag/evaluate.py)

- [ ] Keep benchmark logic unchanged for the first pass so comparisons remain apples-to-apples
- [ ] Run the same eval dataset in all 3 modes:
  - `semantic`
  - `keyword`
  - `hybrid`
- [ ] Save outputs with `--output-dir`

Command:

```bash
uv run python -m arxiv_rag.evaluate data/eval_queries.json --output-dir eval_results/
```

Compare:
- `Recall@5`
- `Recall@10`
- `MRR`
- worst-query list

## Optional Inspection Upgrade

File: [src/arxiv_rag/evaluate.py](/Users/wu36/Code/arXiv-rag-project/src/arxiv_rag/evaluate.py)

- [ ] Extend `format_query_inspection()` to include retrieval source
- [ ] Show whether each hit came from:
  - `semantic`
  - `keyword`
  - `both`

This is optional but strongly recommended for debugging hybrid behavior.

## Tests

File: [tests/test_query_engine.py](/Users/wu36/Code/arXiv-rag-project/tests/test_query_engine.py)

- [ ] Add a test for `retrieval_mode="semantic"`
- [ ] Add a test for `retrieval_mode="keyword"`
- [ ] Add a test for `retrieval_mode="hybrid"`
- [ ] Add a test that hybrid mode deduplicates papers by ID
- [ ] Add a test that a paper returned by both branches gets the expected fused preference
- [ ] Add a test for weight-driven ranking behavior

File: [tests/test_evaluate.py](/Users/wu36/Code/arXiv-rag-project/tests/test_evaluate.py)

- [ ] Add inspection-output coverage if `source` is surfaced
- [ ] Keep existing benchmark artifact tests passing

## Verification Checkpoints

Before tuning:

```bash
uv run pytest tests/test_query_engine.py tests/test_evaluate.py tests/test_repo_surface.py -v
```

After first implementation:

```bash
uv run python -m arxiv_rag.evaluate data/eval_queries.json --output-dir eval_results/
```

Manual inspection:

```bash
uv run arxiv-rag-inspect-query \
  "category learning and human categorization" \
  --expected-id 2403.03835 \
  --expected-id 1304.3432
```

## Tuning Order

- [ ] Run `semantic` baseline
- [ ] Run `keyword` only
- [ ] Run `hybrid` with `0.7 / 0.3`
- [ ] If keyword-heavy queries improve but semantic queries degrade, reduce keyword weight
- [ ] Try `0.6 / 0.4`
- [ ] Try `0.5 / 0.5`
- [ ] Stop tuning once gains flatten

## Success Criteria

- [ ] Hybrid retrieval beats semantic baseline on `Recall@5`
- [ ] Hybrid retrieval does not materially reduce `MRR`
- [ ] Worst-query list gets shorter
- [ ] Manual inspection shows exact-term queries benefiting from keyword retrieval
- [ ] Code remains compatible with existing agent and evaluation entrypoints

## Nice-to-Have Follow-ups

- [ ] Add a reranking pass after hybrid candidate generation
- [ ] Add per-result provenance (`semantic`, `keyword`, `both`)
- [ ] Add hybrid latency reporting to saved eval artifacts
- [ ] Add CSV/markdown comparison across retrieval modes
