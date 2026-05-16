# Hybrid Weight Tuning Design

## Goal

Run a tuning-and-inspection slice for hybrid retrieval to determine whether weight changes alone can close the large gap between current `hybrid` results and `keyword` results on `data/eval_queries.json`.

This slice includes:

- additional hybrid eval runs at new weight settings
- artifact separation for each tuned run
- targeted manual inspection of failing queries
- a decision summary about whether the next step should remain tuning-only or move into fusion-logic changes

This slice does not include:

- changes to retrieval or fusion implementation
- changes to benchmark code
- changes to saved artifact schema
- automatic selection of new production default weights

## Weight Sweep

Starting point already measured:

- `0.7 / 0.3` semantic/keyword

New runs for this slice:

- `0.6 / 0.4`
- `0.5 / 0.5`
- `0.3 / 0.7`

`hybrid_candidate_pool` remains unchanged for this pass so the experiment isolates the effect of weights.

## Evaluation Method

Each tuned run will:

- set `retrieval_mode: "hybrid"`
- update `hybrid_semantic_weight`
- update `hybrid_keyword_weight`
- run the existing evaluation command against `data/eval_queries.json`
- save to a dedicated output directory

Recommended output roots:

- `eval_results/hybrid_06_04`
- `eval_results/hybrid_05_05`
- `eval_results/hybrid_03_07`

## Comparison Targets

Each tuned run will be compared against:

- semantic baseline
- keyword-only baseline
- existing hybrid `0.7 / 0.3` baseline

Primary metrics:

- `Recall@5`
- `Recall@10`
- `MRR`
- miss list size and contents

## Inspection Questions

For the worst failing queries, the inspection should answer:

1. Does keyword-only retrieve the relevant paper in the top results?
2. Does hybrid still miss it entirely, or does it appear but at a lower rank?
3. If it appears but is pushed down, does that point to weight imbalance rather than candidate-generation failure?

The first manual inspection target remains:

- `category learning and human categorization`

Additional inspection targets should come from queries that remain hybrid misses while keyword succeeds.

## Decision Rule

This slice ends with a recommendation, not a code change.

Possible outcomes:

- **Weights are sufficient:** one tuned hybrid setting approaches keyword performance while improving over semantic and the failure pattern looks score-weight related
- **Weights are not sufficient:** keyword still clearly outperforms all hybrid settings, or relevant hits are absent from hybrid top candidates
- **Mixed result:** weights help, but remaining misses suggest a second pass on candidate pool or fusion method

## Risks And Constraints

Because no retrieval code changes are made in this slice, poor outcomes should be interpreted as evidence about the current fusion strategy rather than implementation instability.

The biggest risk is over-reading small metric changes on an 11-query eval set, so the summary should emphasize directionality and query-level behavior, not just single-number winners.
