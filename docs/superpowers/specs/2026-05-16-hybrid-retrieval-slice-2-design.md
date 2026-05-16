# Hybrid Retrieval Slice 2 Design

## Goal

Implement the first production-ready hybrid retrieval slice by adding keyword and hybrid retrieval modes to the existing query engine while preserving the current public search interface used by the agent tooling.

This slice includes:

- retrieval config additions in `config.yaml`
- `QueryEngine` refactor behind the existing `search()` entrypoint
- hybrid late-fusion ranking
- unit coverage for retrieval modes and agent compatibility

This slice does not include:

- evaluation output formatting changes
- retrieval tuning beyond the initial default weights
- new public retrieval APIs

## Public Interface

`QueryEngine.search(query, filters=None, limit=None)` remains the only public retrieval API.

Callers continue to receive a list of dictionaries. The stable keys required by downstream callers remain:

- `id`
- `title`
- `authors`
- `abstract`
- `categories`

The query engine will also attach:

- `score`
- `source`

The agent tooling will continue to ignore `score` and `source` for now, keeping backward compatibility with its existing normalization flow.

## Retrieval Modes

`config.yaml` will gain the following settings:

- `retrieval_mode`, defaulting to `"semantic"`
- `hybrid_semantic_weight`, defaulting to `0.7`
- `hybrid_keyword_weight`, defaulting to `0.3`
- `hybrid_candidate_pool`, defaulting to `25`

`search()` behavior:

- `semantic`: call `_search_semantic(query, limit)`
- `keyword`: call `_search_keyword(query, limit)`
- `hybrid`: call both branch searches with the configured candidate pool, normalize branch scores independently, fuse by paper ID, and return the top requested results

Unsupported retrieval modes will raise `ValueError` with the invalid mode name.

## Query Engine Structure

`QueryEngine` will be refactored to use private helpers:

- `_search_semantic(query, limit)`
- `_search_keyword(query, limit)`
- `_normalize_scores(rows)`
- `_fuse_results(semantic_rows, keyword_rows, semantic_weight, keyword_weight)`

The semantic helper will preserve current behavior:

- encode and normalize the query embedding
- rank with pgvector cosine distance
- convert distance to similarity score
- dedupe repeated paper IDs before truncating to the requested limit

The keyword helper will:

- search against a combined text vector derived from title and abstract
- require `to_tsvector(...) @@ plainto_tsquery(...)`
- compute ranking with `ts_rank_cd(...)`
- sort by keyword relevance descending
- return the same row shape as the semantic branch

## Scoring And Fusion

Each branch emits rows with raw branch-local `score` values and a branch-specific `source` value:

- semantic rows use `source="semantic"`
- keyword rows use `source="keyword"`

`_normalize_scores(rows)` will convert branch scores to the `0..1` range independently per branch.

Normalization rules:

- empty input returns an empty list
- if all scores are identical and the list is non-empty, normalized score becomes `1.0` for every row
- otherwise use min-max normalization across the branch

`_fuse_results(...)` will:

- merge rows by paper ID
- treat missing semantic or keyword branch scores as `0`
- compute fused score as `(normalized_semantic * semantic_weight) + (normalized_keyword * keyword_weight)`
- mark `source="both"` when a paper appears in both branches
- otherwise preserve the originating branch source
- sort by fused score descending

When duplicate metadata appears for the same paper ID, the fused row will keep the first available paper fields from the branch rows and only combine scores and provenance.

## Agent Compatibility

`RetrievalTool.search()` will continue to call `QueryEngine.search()` without interface changes.

Compatibility requirements:

- rows must still include `id` and `title`
- rows must continue to provide usable `authors`, `abstract`, and `categories`
- extra keys must not be required by the tool layer

No `RetrievalTool` API changes are required in this slice.

## Testing

The first pass will add unit tests for:

- semantic mode routing
- keyword mode routing
- hybrid mode routing
- hybrid deduplication by paper ID
- fused ranking preference when a paper appears in both branches
- weight-driven ranking changes
- retrieval tool compatibility with rows that include `score` and `source`

The tests will avoid requiring a live database by stubbing query-engine internals where appropriate and focusing on branch composition and return shapes.

## Risks And Constraints

The main compatibility risk is changing row shape or ranking semantics in ways that break the existing agent or evaluation paths. Keeping `search()` stable and adding only backward-compatible keys minimizes this risk.

Keyword-only queries may produce score ranges very different from semantic similarity scores, so hybrid mode must normalize scores per branch before fusion. Raw-score fusion is explicitly out of scope because it would bias the combined ranking toward whichever branch has the larger numeric scale.
