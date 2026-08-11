# Member 4 → Member 5 Handoff

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)
Recipient: Member 5 (FastAPI + API Integration)
Covers the Section 8 checklist for this handoff, plus the Gate 3 and Gate 4 items
from Section 9.

## Consume

One entry point:

```python
from src.evaluation.engine import evaluate_batch_safe

result = evaluate_batch_safe(qa_pairs)
```

- `qa_pairs`: list of `{"task": str, "student_prompt": str}` — optionally with
  `"domain"` and `"level"` to scope retrieval to a metadata slice.
- Returns either the evaluation result or `{"error": {...}}`. It does not raise
  for a pipeline failure, so `/evaluate` needs no broad `except`.
- `evaluate_batch` is the same function with typed exceptions
  (`PecsEvaluationError`) if you prefer that style.

Full field-by-field response contract, the error-code → HTTP-status table, and the
layer boundaries: **`docs/evaluation_architecture.md`**. Please read it before
starting API-001 — the tracker makes your review the acceptance criterion for
EVAL-014.

## Handoff checklist (Section 8)

- [x] `retrieve_context()` and `build_batch_content()` functioning correctly —
      `src/rag/retrieve_context.py`, `src/rag/build_batch_content.py`, 35 + 30
      passing tests.
- [x] System prompt encodes the current C1–C6 rubric with caching enabled —
      `src/rag/system_prompt.py`; `cache_control: {"type": "ephemeral"}` set in
      `src/rag/build_request.py`. **Caveat below — the marker is set but will not
      produce cache hits on Haiku 4.5.**
- [x] `calculate_scores()` and skill-level mapping verified against hand-computed
      cases — 8/7/6/7/9/5 → 7.05, all-10s → 10.0 (not 100), plus a parity test
      against Member 2's ANN-004 `weighted_score()`.
- [x] C5 ethics handling behaviour implemented — **weighted-only in this build;
      the policy decision is still unconfirmed, see below.**
- [x] Evaluation architecture documentation complete —
      `docs/evaluation_architecture.md`, `docs/evaluator_determinism.md`.
- [ ] Handoff notes acknowledged by Member 5 — pending your sign-off.

## What you own that I deliberately did not do

- **Pydantic request/response models (API-002/003).** The engine returns plain
  JSON-serialisable dicts. I did not define Pydantic models, because the wire
  contract is yours to version with the Question Engine and Assessment Engine
  (INT-001…INT-003).
- **HTTP status mapping (API-010).** I supply `error.code`, `error.layer` and
  `error.retryable`; `docs/evaluation_architecture.md` proposes a mapping table,
  but the final choice is yours.
- **Batch-size limits (API-006).** The engine enforces a *token* ceiling
  (RAG-019, ~4,000 estimated input tokens) and aborts before calling the model. It
  does not enforce a submission-count limit — decide that at the API edge where
  you can return a clean 4xx before any work happens.
- **Auth, rate limiting, request logging.** Untouched by this layer. Note the
  engine already emits one `pecs.rag.usage` JSON log line per completed
  evaluation (RAG-020); route that logger wherever your logging config sends
  records, and avoid double-logging token counts.

Two behaviours worth knowing before you write the error handler:

- **Not every failure is retried internally.** RAG-017 retries an unusable
  response body (bad JSON, schema violation, empty response) up to
  `PECS_MAX_LLM_ATTEMPTS`. It does **not** retry `RAG_LLM_REFUSAL` (deterministic
  policy outcome), transport failures (the SDK already retries those), or
  configuration errors. So a `RAG_LLM_REFUSAL` reaching you means one call was
  made, not three — do not add your own retry on that code.
- **Per-submission records are self-describing.** Each entry in `evaluations`
  carries its own `model_version`, `rubric_version` and `rubric_tag` in addition
  to the batch-level copies, so you can persist one row per submission without
  denormalising the batch fields yourself.

## Configuration you will need

Set in `.env` (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Evaluator credential. **Required** — without it the first call returns `RAG_LLM_AUTH_ERROR`. | — |
| `CHROMA_DB_PATH` | Vector store location (Member 3's VDB-001). | `Database/pels_vector_db` |
| `PECS_EVALUATOR_MODEL` | Evaluator model. | `claude-haiku-4-5-20251001` |
| `PECS_RAG_TOP_K` | Calibration examples per prompt. | `1` |
| `PECS_TOKEN_SAFETY_CAP` | Pre-flight input-token ceiling. | `4000` |
| `PECS_MAX_PROMPT_CHARS` | Per-submission truncation limit. | `1000` |
| `PECS_MAX_LLM_ATTEMPTS` | Retry budget for malformed responses. | `3` |
| `PECS_SCORE_RANGE_MODE` | `reject` or `clamp` out-of-range scores. | `reject` |
| `PECS_ETHICS_HARD_GATE` | C5 hard-gate veto on/off. | `false` |

Every value is read at import time by `src/rag/config.py`; `config.snapshot()`
returns the effective set and is embedded in each result under `meta.config`.

## Performance notes for the API layer

- **One API call per request**, no matter the batch size — that is the point of
  the batched design. Grading 3 submissions costs one call, not three.
- **First request is slow.** The sentence-transformer model (EMB-001) warm-loads
  on first use, which takes seconds. Trigger it during FastAPI startup rather
  than inside the first user request, e.g.
  `from src.embeddings.embedder import get_embedder; get_embedder()` in a startup
  hook. `/health` (API-004) is a good place to assert it is loaded.
- **The engine is synchronous.** The Anthropic call blocks. Under `async def`
  endpoints, run it in a threadpool (`fastapi.concurrency.run_in_threadpool`) or
  declare the endpoint `def` so Starlette does that for you — otherwise one slow
  evaluation blocks the event loop.
- **Latency is dominated by the model call.** Retrieval is a single-vector
  ChromaDB query; assembly and scoring are pure Python.

## Two items I am handing over honestly rather than quietly

**1. C5 ethics handling is unconfirmed policy (EVAL-008, `[DATA NEEDED]`).**
The tracker requires written confirmation from Team Beta / Alliance Compliance on
whether C5 acts as a hard-gate veto in this MVP. It has not arrived, so the build
runs **weighted-only** and every result carries
`ethics_gate.policy_confirmation = "pending_alliance_compliance_confirmation"`.
The hard-gate path is implemented and tested behind `PECS_ETHICS_HARD_GATE`.
Please surface `ethics_gate` in the API response so downstream teams can see which
policy produced a score, and flag the outstanding confirmation in INT-006/INT-007.

**2. Prompt caching will not take effect on Haiku 4.5 (RAG-012).**
The implementation is correct — ephemeral `cache_control` on the rubric block,
volatile batch content kept out of the cached prefix — but Claude Haiku 4.5 has a
**4,096-token minimum cacheable prefix** and the condensed rubric is well under
it. The API silently skips caching a shorter prefix, so
`cache_read_input_tokens` will be `0`. `meta.caching` reports this explicitly:

```json
{"requested": true, "prefix_tokens": 900, "minimum_tokens": 4096,
 "expected_effective": false}
```

Do not report a cache-hit rate in the API docs until the team decides between
lengthening the cached prefix past 4,096 tokens or moving to a model with a lower
minimum. Both change cost and behaviour, so neither was done unilaterally.

## Also worth knowing

- **`Database/` is created by importing the vector-store package.** Member 3's
  `client.py` builds a `PersistentClient` at import time, so the directory appears
  as soon as anything imports `src.vectorstore.*`. It has been added to
  `.gitignore` (which FOUND-003 already called for). A separate task has been
  filed to make that client lazy; it is not a blocker for you.
- **Test doubles are reusable.** `tests/member4/conftest.py` has `FakeCollection`,
  `FakeEmbedder`, `FakeClient`, `FakeResponse` and evaluator-payload builders.
  Use them for API integration tests (API-014) so your tests do not need a live
  key: pass `client=FakeClient([...])` straight through to the engine.
- **`tests/member4/` needs no infrastructure** — no ChromaDB, no embedding model,
  no API key. `pytest tests/member4 -q` → 187 passing.

## For Member 6 (forwarded through you)

- BENCH-004's harness can call `evaluate_batch(..., retrieve_calibration=False)`
  to measure uncalibrated grading as a baseline against the calibrated run.
- BENCH-007 (consistency) must hold batch composition constant between repeats —
  submissions share one call, so the batch is part of the context. Rationale in
  `docs/evaluator_determinism.md`.
- Aggregate cost per benchmark run from the `pecs.rag.usage` log lines; each line
  is one completed evaluation with token counts and an estimated USD cost.
