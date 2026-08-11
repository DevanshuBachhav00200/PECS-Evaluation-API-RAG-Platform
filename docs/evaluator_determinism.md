# Evaluator Determinism Configuration

**Task:** EVAL-012 — Review deterministic-settings configuration
**Owner:** Member 4 (RAG Pipeline + AI Evaluation Engine)
**Status:** Reviewed, documented, rationale stated.

## The configuration in force

| Setting | Value | Where |
|---|---|---|
| `temperature` | `0.0` | `config.TEMPERATURE` (`PECS_EVALUATOR_TEMPERATURE`) |
| `top_p` / `top_k` | not sent | — |
| `model` | `claude-haiku-4-5-20251001` (dated snapshot) | `config.MODEL_ID` (`PECS_EVALUATOR_MODEL`) |
| `max_tokens` | `4096` | `config.MAX_TOKENS` |
| System prompt | frozen constant, assembled at import | `src/rag/system_prompt.py` |
| Retrieval Top-K | `1` | `config.DEFAULT_TOP_K` |

## Rationale

**`temperature = 0.0`.** Scoring is a classification task against a fixed rubric,
not a generation task — we want the same prompt to receive the same score twice.
Temperature 0 gives the most repeatable sampling available on this model.

**`top_p` / `top_k` are not sent at all.** Setting one of them alongside
temperature adds a second, independent sampling control for no benefit here, and
on newer Claude models sampling parameters are rejected outright. Omitting them
keeps the request valid across model upgrades.

**Dated model snapshot rather than the `claude-haiku-4-5` alias.** An alias can
be repointed at a newer snapshot, which would silently change scoring behaviour
mid-benchmark. Pinning the dated ID means a model change is a deliberate config
edit, and EVAL-013 records the exact string on every result.

**Frozen system prompt.** No timestamps, UUIDs, or per-request interpolation in
the rubric block. This is required twice over: for repeatability, and because
prompt caching is a prefix match (RAG-012) — one varying byte would both change
the model's input and destroy the cache.

## What temperature 0 does *not* buy us

Be precise about this in the benchmark report, because it is easy to overclaim:

- **Temperature 0 is not a determinism guarantee.** It makes the model pick the
  highest-probability token at each step, but floating-point non-associativity in
  batched inference, server-side model revisions, and load-dependent kernel
  selection all mean two identical requests can still differ. Anthropic does not
  document bitwise reproducibility at any temperature.
- **Consistency must therefore be measured, not assumed.** BENCH-007
  (consistency testing) should run the same submission N times and report the
  observed score variance. That measurement is the evidence; this configuration
  only maximises the chance that it comes out small.
- **Prompt caching does not affect the output distribution.** A cache read
  changes how the prefix is billed, not what the model computes from it.

## Sources of run-to-run variation we *do* control

| Source | Control | Notes |
|---|---|---|
| Retrieved calibration example | Top-K = 1, deterministic embeddings (EMB-005) | The same prompt retrieves the same example while the corpus is unchanged. Re-ingesting the corpus can change scores — that is a corpus version change, not evaluator drift. |
| Rubric text | `rubric_tag` on every result (EVAL-003) | A rubric edit changes the tag, so mixed-rubric results are visible in the data. |
| Model | `model_version` on every result (EVAL-013) | Prevents comparing across model revisions unknowingly. |
| Batch composition | Documented, not eliminated | See below. |

## Known non-determinism: batch composition

Submissions are graded in one batched call, so the other submissions in the batch
are part of the model's context. The same prompt evaluated alone and evaluated
alongside five stronger prompts can receive a different score.

This is inherent to the batched design (which is what makes one call per batch
possible) and is **not** a bug, but it must be held constant when measuring:

- BENCH-004 should use a fixed batching for the gold set, and record it.
- BENCH-007's consistency runs should keep batch composition identical between
  repetitions, otherwise the measurement conflates sampling variance with
  batch-context variance.

If a future revision needs per-submission isolation, set the batch size to 1 —
cost rises roughly linearly, which is the trade being made.

## Review outcome

The configuration is appropriate for the MVP: maximally repeatable settings, all
sources of drift either pinned or recorded on the result, and the residual
variance sources documented above so the benchmark reports them honestly rather
than claiming determinism the platform does not offer.
