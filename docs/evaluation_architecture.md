# Evaluation Architecture — Separation of Concerns

**Task:** EVAL-014 — Document separation of concerns across the evaluation stack
**Owner:** Member 4 (RAG Pipeline + AI Evaluation Engine)
**Audience:** Member 5 (FastAPI integration), Member 6 (testing/benchmarking), future maintainers
**Review gate:** must be read by Member 5 before FastAPI integration begins.

## The four layers, and what each one may not do

```
        ┌──────────────────────────────────────────────────────────┐
   (1)  │ RETRIEVAL          normalize → embed → query → format     │
        │                    src/rag/query_preprocess, query_embed,  │
        │                    retrieve_context, filtered_context,     │
        │                    format_calibration, empty_retrieval     │
        └───────────────────────────┬──────────────────────────────┘
                                    │  compact calibration lines
        ┌───────────────────────────▼──────────────────────────────┐
   (2)  │ LLM EVALUATION     assemble → pre-flight → call → parse    │
        │                    → validate → retry                     │
        │                    src/rag/build_batch_content, truncate,  │
        │                    system_prompt, build_request, llm_call,  │
        │                    token_safety, parse_response,            │
        │                    validate_llm_output, retry_handler,      │
        │                    api_error_handler, usage_logger          │
        └───────────────────────────┬──────────────────────────────┘
                                    │  structurally valid C1–C6 JSON
        ┌───────────────────────────▼──────────────────────────────┐
   (3)  │ SCORE COMPUTATION  range guard → composite → skill level   │
        │                    → ethics handling → justification caps  │
        │                    src/evaluation/score_range_guard,        │
        │                    calculate_scores, skill_level_map,        │
        │                    ethics_gate, justification,               │
        │                    overall_summary, confidence               │
        └───────────────────────────┬──────────────────────────────┘
                                    │  engine result dict
        ┌───────────────────────────▼──────────────────────────────┐
   (4)  │ API RESPONSE FORMATTING            Member 5 — API-002/003  │
        │                    Pydantic models, HTTP status mapping,    │
        │                    auth, rate limiting, request logging     │
        └──────────────────────────────────────────────────────────┘
```

`src/evaluation/engine.py` sequences layers 1–3. It contains no scoring
arithmetic, no prompt text, and no HTTP concepts — only ordering.

### The boundaries that matter

| Layer | Owns | Must **not** |
|---|---|---|
| 1 Retrieval | Turning a prompt into a calibration line; deciding there is no usable example | Score anything; know the rubric weights; construct an API request |
| 2 LLM evaluation | The prompt contract, one call per batch, transport failures, response *shape* | Compute a composite; enforce score ranges; know HTTP |
| 3 Score computation | Ranges, weights, bands, ethics policy, word caps, version tags | Call the model; retry; touch the vector store |
| 4 API (Member 5) | Pydantic models, status codes, auth, rate limits | Re-derive scores; re-validate the rubric; reformat justifications |

Two rules keep these honest and are worth preserving under future change:

1. **The rubric has exactly one source.** Weights come from
   `src/dataset/annotation/annotation_schema.py` (ANN-001), bands from
   `skill_level_map.py`, prompt text from `system_prompt.py`, and everything
   loads through `rubric_loader.py` (EVAL-002). Nothing restates a weight.
2. **Composite scoring is not the model's job.** The evaluator returns six scores
   and six justifications. The composite, the band, and the ethics decision are
   computed in code, so they are auditable, testable against hand calculations,
   and cannot drift with model behaviour.

## What Member 5 calls

```python
from src.evaluation.engine import evaluate_batch_safe

result = evaluate_batch_safe(qa_pairs)   # never raises for a pipeline failure

if "error" in result:
    raise HTTPException(status_code=STATUS_MAP[result["error"]["code"]],
                        detail=result["error"])
```

`evaluate_batch` is the same call with exceptions instead of an error object; use
whichever fits the endpoint style. Every injectable dependency
(`collection`, `embedder`, `client`, `retriever`) defaults to the real thing, so
production callers pass only `qa_pairs`.

### Input

`qa_pairs` is a sequence; each element is either

```python
{"task": "...", "student_prompt": "...", "domain": "CODE", "level": "strong"}
```

(`domain` / `level` optional — when given, retrieval is scoped to them) or a
`(task, student_prompt)` tuple.

### Output

```python
{
  "evaluations": [
    {
      "submission_index": 1,
      "task": "...",
      "scores": {"c1": 8.0, ..., "c6": 5.0},        # floats, 1–10, validated
      "composite_score": 7.05,                       # 0–10, weighted, no ×10
      "skill_level": "strong",                       # weak | developing | strong
      "ethics_gate": {"mode": "weighted_only", "triggered": False, ...},
      "justifications": {"c1": "...", ..., "c6": "..."},
      "justification_truncated": [],                 # criteria whose text was cut
      "prompt_truncated": False,                     # RAG-010 fired?
      "calibration_used": True,
      "calibration_record_id": "rec-001",
      "domain": None, "level": None,
      "confidence": None,                            # EVAL-010, absent in MVP
      "confidence_label": None,
      "confidence_status": "not_implemented_in_mvp",
      "model_version": "claude-haiku-4-5-20251001",  # EVAL-013, per record
      "rubric_version": "pecs-rubric-v1.0",          # EVAL-003, per record
      "rubric_tag": "pecs-rubric-v1.0+<fingerprint>"
    }
  ],
  "overall_summary": {
    "strength": "...", "weakness": "...", "improvement": "...",
    "source": "evaluator",                           # evaluator | derived | mixed
    "truncated": [],
    "criterion_averages": {"c1": 8.0, ...}
  },
  "usage": {
    "input_tokens": 1200, "output_tokens": 300,
    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    "attempts": 1, "estimated_input_tokens": 900, "cost_usd": 0.0027
  },
  "rubric_version": "pecs-rubric-v1.0",
  "rubric_tag": "pecs-rubric-v1.0+<fingerprint>",
  "model_version": "claude-haiku-4-5-20251001",
  "meta": {"submissions": 1, "calibrated_submissions": 1, "weights": {...},
           "config": {...}, "caching": {...}}
}
```

The whole dict is JSON-serialisable (asserted by a test) so it can be returned
directly or mapped through Pydantic response models. `model_version`,
`rubric_version` and `rubric_tag` appear both at batch level and on every
per-submission record, so a stored row is self-describing for Member 6's
benchmark tables without the caller copying the batch-level fields down.

### Error object

```python
{"error": {"code": "RAG_TOKEN_SAFETY_ABORT", "layer": "rag",
           "message": "SAFETY ABORT: estimated input tokens (5200) exceed ...",
           "details": {...}, "retryable": False}}
```

`layer` is `rag` or `evaluation`; `retryable` is set for rate limits, timeouts,
connection failures and 5xx. Messages never contain a stack trace. Suggested
mapping for API-010 (Member 5 owns the final choice):

| Code | HTTP |
|---|---|
| `EVAL_INVALID_INPUT` | 422 |
| `RAG_TOKEN_SAFETY_ABORT` | 413 |
| `RAG_LLM_AUTH_ERROR` | 502 (our credential, not the caller's) |
| `RAG_LLM_RATE_LIMITED` | 429 |
| `RAG_LLM_TIMEOUT`, `RAG_LLM_CONNECTION_ERROR`, `RAG_LLM_SERVER_ERROR` | 502 / 504 |
| `RAG_LLM_REFUSAL` | 502 — evaluator declined; **not retried**, re-sending identical input cannot change a policy outcome |
| `RAG_LLM_EMPTY_RESPONSE` | 502 — only seen after the retry budget is exhausted |
| `RAG_RESPONSE_*`, `RAG_RETRY_BUDGET_EXHAUSTED` | 502 |
| `EVAL_SCORE_OUT_OF_RANGE`, `EVAL_MISSING_CRITERION`, `EVAL_SUBMISSION_COUNT_MISMATCH` | 502 (evaluator misbehaved, caller's request was fine) |
| `EVAL_RUBRIC_UNAVAILABLE` | 500 |

## Scoring formula (the one that must not regress)

```
composite = Σ score[c] × weight[c]     over c1..c6
weights   = C1 15% · C2 20% · C3 20% · C4 20% · C5 15% · C6 10%   (sum = 1.0)
bands     = weak 0.00–3.99 · developing 4.00–6.99 · strong 7.00–10.00
```

There is **no ×10 multiplication.** All-10s gives `10.0`, not `100`. Two tests
pin this: a hand-calculated case (8/7/6/7/9/5 → 7.05) and a parity check against
Member 2's ANN-004 `weighted_score()`.

## Two open items carried forward honestly

**1. C5 ethics handling is weighted-only, pending confirmation (EVAL-008).**
The tracker flags `[DATA NEEDED]`: written confirmation from Team Beta /
Alliance Compliance on whether the hard-gate veto is active in this build. It has
not been received, so the build runs **weighted-only** and every result records
`ethics_gate.policy_confirmation = "pending_alliance_compliance_confirmation"`.
The hard-gate path is fully implemented and tested; switching it on is
`PECS_ETHICS_HARD_GATE=true` plus a rubric-version bump. Do not report gated and
ungated results in the same benchmark table.

**2. Prompt caching is requested but will not take effect on Haiku 4.5 yet
(RAG-012).** The system block carries `cache_control: {"type": "ephemeral"}`, and
volatile content is kept out of the cached prefix, so the *implementation* is
correct. But Claude Haiku 4.5 has a **4,096-token minimum cacheable prefix**, and
the condensed PECS rubric is well under it — the API silently declines to cache a
shorter prefix (no error, `cache_creation_input_tokens: 0`). So RAG-012's
acceptance criterion ("a second call within the cache window shows a cache-read
hit") will **not** be met on this model with this prompt, through no fault of the
code.

`caching_diagnostics()` reports this explicitly, and `meta.caching` carries it on
every result:

```python
{"requested": True, "prefix_tokens": 900, "minimum_tokens": 4096,
 "expected_effective": False}
```

Options when the team wants real cache hits: put more stable content in the cached
prefix (full rubric anchors, few-shot examples) to clear 4,096 tokens; or move the
evaluator to a model with a lower minimum. Both are cost/quality decisions for the
team, not silent code changes — hence this note rather than a padded prompt.

## Testing the stack without infrastructure

Every layer is dependency-injected, so `tests/member4/` runs with no ChromaDB, no
embedding model, and no API key: fakes live in `tests/member4/conftest.py`
(`FakeCollection`, `FakeEmbedder`, `FakeClient`, `FakeResponse`). Member 6 can
reuse them for TEST-004/TEST-005/TEST-006 rather than writing new doubles.

```bash
pytest tests/member4 -q      # 187 passed
```

The only tests that need real infrastructure are the end-to-end ones Member 6
owns (TEST-007) and the benchmark harness (BENCH-004), which need a populated
vector store and a live API key.
