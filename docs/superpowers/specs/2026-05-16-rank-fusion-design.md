# Rank Fusion Hybrid Retrieval Design

## Goal

Replace the current raw-score hybrid fusion with rank-based fusion so keyword-successful queries are less likely to be suppressed by semantic score scaling.

This slice includes:

- a hybrid-only fusion logic change in `src/arxiv_rag/query_engine.py`
- test updates in `tests/test_query_engine.py`
- re-running the evaluation benchmark to compare the new hybrid against prior baselines

This slice does not include:

- changes to pure semantic retrieval
- changes to pure keyword retrieval
- changes to the `QueryEngine.search()` public interface
- changes to evaluation artifact schema

## Motivation

The weight-tuning slice showed that:

- keyword-only retrieval strongly outperforms semantic on the current eval set
- weight tuning helps hybrid a lot
- even the best tuned setting still suppresses some strong keyword hits

That pattern suggests the current issue is not just the balance of weights, but the use of normalized raw branch scores as the fusion basis.

## Proposed Fusion Method

Hybrid mode will continue to fetch semantic and keyword candidate pools separately, dedupe by paper ID, and return the same unified row shape.

The difference is how fused scores are computed.

Instead of:

- normalizing raw semantic scores
- normalizing raw keyword scores
- summing weighted normalized values

the new method will use reciprocal-rank-style fusion:

- semantic contribution: `semantic_weight / (rank_constant + semantic_rank)`
- keyword contribution: `keyword_weight / (rank_constant + keyword_rank)`
- missing branch contribution: `0`
- fused score: sum of available branch contributions

Where:

- `semantic_rank` and `keyword_rank` are 1-based ranks within each branch result list
- `rank_constant` is a small positive config or internal constant that controls how sharply top ranks dominate

## Query Engine Shape

The public search API remains:

- `QueryEngine.search(query, filters=None, limit=None)`

Semantic and keyword branch helpers continue returning:

- `id`
- `title`
- `authors`
- `abstract`
- `categories`
- `score`
- `source`

Hybrid fused rows will continue returning the same shape and provenance rules:

- `source="both"` when the paper appears in both branches
- otherwise preserve `semantic` or `keyword`

## Internal Refactor

`QueryEngine` will stop using normalized raw scores for hybrid ranking.

Expected internal shape:

- keep `_search_semantic(...)`
- keep `_search_keyword(...)`
- replace `_normalize_scores(...)` usage in hybrid mode
- introduce a rank-based helper such as `_fuse_ranked_results(...)`

If `_normalize_scores(...)` is no longer needed anywhere, it can be removed along with its tests. If keeping it reduces churn, it may remain unused temporarily, but the preferred outcome is to align tests and code with the actual fusion strategy.

## Testing

`tests/test_query_engine.py` should be updated to reflect rank-based behavior.

Coverage goals:

- hybrid still routes correctly
- deduplication by paper ID still works
- a paper appearing in both branches gets combined preference
- rank-based weighting can let a strong keyword rank outrank a weaker semantic-only result
- ranking behavior no longer depends on raw branch score magnitudes

## Evaluation

After implementation, rerun:

- hybrid with the chosen rank-fusion defaults

Compare against the existing stored baselines for:

- semantic
- keyword
- score-fused hybrid `0.7 / 0.3`
- tuned score-fused hybrid `0.3 / 0.7`

Primary metrics:

- `Recall@5`
- `Recall@10`
- `MRR`
- miss list

## Risks And Constraints

Rank fusion usually improves robustness when branch score scales are incomparable, but it also discards branch score magnitude within a rank. The new tests and eval rerun are important because this may help exact-match queries while changing behavior on semantic-heavy queries.

The slice should stay focused on the fusion method itself. Candidate-pool expansion or reranking rules should be treated as later follow-ups unless the evaluation shows rank fusion alone is still not enough.
