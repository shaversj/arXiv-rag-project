# Hybrid Eval Slice Design

## Goal

Implement the next hybrid-retrieval evaluation slice by adding provenance to query inspection output and then running the labeled evaluation dataset in all three retrieval modes for an apples-to-apples comparison.

This slice includes:

- a small inspection-only enhancement in `src/arxiv_rag/evaluate.py`
- focused test coverage in `tests/test_evaluate.py`
- three benchmark runs against `data/eval_queries.json`
- a comparison of `Recall@5`, `Recall@10`, `MRR`, and miss lists across modes

This slice does not include:

- changes to benchmark scoring logic
- changes to saved artifact schema
- weight tuning beyond the initial `0.7 / 0.3` hybrid default
- additional comparison report formats

## Evaluation Behavior

`evaluate_queries()` and the benchmark artifact-writing flow remain unchanged so comparisons stay stable across retrieval modes.

The only code-path change is in `format_query_inspection(...)`, which will optionally include retrieval provenance when a result row includes `source`.

Compatibility rules:

- rows without `source` continue to render exactly as before
- rows with `source` render the source inline with the displayed hit
- supported provenance values for this slice are `semantic`, `keyword`, and `both`

## Output Shape

Current inspection output includes rank, paper ID, and title. After this slice:

- rows without `source`: `1. <id> | <title>`
- rows with `source`: `1. <id> | <title> | source=<source>`

This keeps the output human-readable while preserving backward compatibility for existing tests and callers.

## Test Coverage

`tests/test_evaluate.py` will gain focused inspection coverage for provenance display.

Coverage goals:

- existing inspection behavior still passes when `source` is absent
- provenance is displayed when `source` is present
- benchmark artifact tests remain unchanged and green

## Benchmark Execution

After the inspection change is merged, the evaluation dataset will be run three times:

- `semantic`
- `keyword`
- `hybrid`

Each run will use the same command shape:

```bash
uv run python -m arxiv_rag.evaluate data/eval_queries.json --output-dir <mode-specific-dir>
```

The retrieval mode will be switched in `config.yaml` between runs. Outputs will be kept in separate directories to avoid mixing artifacts.

Recommended output roots:

- `eval_results/semantic`
- `eval_results/keyword`
- `eval_results/hybrid`

## Comparison And Inspection

The comparison summary for this slice will focus on:

- `Recall@5`
- `Recall@10`
- `MRR`
- worst-query / miss list

One manual inspection run will also be used for sanity checking provenance on the known category-learning query:

```bash
uv run arxiv-rag-inspect-query \
  "category learning and human categorization" \
  --expected-id 2403.03835 \
  --expected-id 1304.3432
```

## Risks And Constraints

The main risk is accidentally changing benchmark semantics while adding provenance display. Keeping the change confined to `format_query_inspection(...)` avoids that.

The benchmark runs may be time-consuming depending on the local database state, but they do not require code changes beyond this slice’s small inspection upgrade.
