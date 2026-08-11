# Member 4 — RAG Pipeline + AI Evaluation Engine (Implementation Notes)

Workstreams F (RAG-001…RAG-020) and G (EVAL-001…EVAL-014), plus the co-owned
RAG/evaluation half of HAND-006.

- Task → file → test index: [`TASK_MAP.md`](TASK_MAP.md)
- Architecture and API contract: [`../evaluation_architecture.md`](../evaluation_architecture.md)
- Determinism review: [`../evaluator_determinism.md`](../evaluator_determinism.md)
- Handoff to Member 5: [`member5_handoff.md`](member5_handoff.md)

## What was built

```
src/rag/                          Workstream F
  config.py                       every tunable, env-overridable (RAG-004)
  query_preprocess.py             query-time normalization (RAG-001)
  query_embed.py                  warm-model query embedding (RAG-002)
  retrieve_context.py             closest calibration example (RAG-003)
  filtered_context.py             domain/level-scoped retrieval (RAG-005)
  empty_retrieval_handler.py      structured 'no calibration' result (RAG-006)
  context_ranking.py              future multi-candidate ranking (RAG-007, off)
  format_calibration.py           compact calibration line (RAG-008)
  build_batch_content.py          batched user message (RAG-009)
  truncate.py                     per-submission truncation (RAG-010)
  system_prompt.py                PECS_SYSTEM_PROMPT (RAG-011)
  build_request.py                Messages API payload + caching (RAG-012/013)
  llm_call.py                     single batched call (RAG-014)
  parse_response.py               fence-stripped JSON parsing (RAG-015)
  validate_llm_output.py          C1-C6 schema validation (RAG-016)
  retry_handler.py                bounded retries (RAG-017)
  api_error_handler.py            transport failure classification (RAG-018)
  token_safety.py                 pre-flight estimate + abort (RAG-019)
  usage_logger.py                 token + cost audit line (RAG-020)

src/evaluation/                   Workstream G
  schema.py                       evaluator input contract (EVAL-001)
  rubric_loader.py                single versioned rubric source (EVAL-002)
  rubric_version.py               version + content fingerprint (EVAL-003)
  score_range_guard.py            1-10 enforcement (EVAL-004)
  justification.py                per-criterion word caps (EVAL-005)
  calculate_scores.py             weighted composite (EVAL-006)
  skill_level_map.py              band mapping (EVAL-007)
  ethics_gate.py                  C5 handling, both modes (EVAL-008)
  overall_summary.py              batch strengths/gaps (EVAL-009)
  confidence.py                   optional confidence field (EVAL-010)
  error_types.py                  layered structured errors (EVAL-011)
  model_version.py                model tracking (EVAL-013)
  engine.py                       the composed engine Member 5 calls

tests/member4/                    187 tests, no infrastructure required
  conftest.py                     fakes reusable by Member 6
  test_retrieval_context.py       RAG-001…RAG-008
  test_prompt_assembly.py         RAG-009…RAG-013, RAG-019
  test_llm_layer.py               RAG-014…RAG-018, RAG-020
  test_evaluation_scoring.py      EVAL-001…EVAL-013
  test_engine.py                  end-to-end (Gate 3 + Gate 4)
```

## Design decisions worth knowing

**Nothing restates the rubric.** Weights come from Member 2's ANN-001
`WEIGHTS`, bands from `skill_level_map`, prompt text from `system_prompt`, and
everything is composed by `rubric_loader` (EVAL-002). `calculate_scores` reads
weights through the loader, so editing the canonical weights changes scoring on
the next call with no edit anywhere else. `skill_level_map` and
`calculate_scores` delegate to Member 2's ANN-004/ANN-005 functions rather than
re-implementing them, and a test asserts the two agree — that is how "matching
ANN-005" is kept true rather than merely intended.

**The model returns six scores; code computes everything else.** Composite,
band, and ethics decision are pure functions, so they are testable against hand
calculations and cannot drift with model behaviour. The system prompt explicitly
forbids the model from computing a composite or a skill level.

**Everything is dependency-injected.** `collection`, `embedder`, `client` and
`retriever` all default to the real implementations and can be replaced. This is
why the whole suite runs in ~3 seconds with no ChromaDB, no torch, and no API
key — and why Member 6 can write TEST-004/005/006 without infrastructure.

**Failures are values, not tracebacks.** Every failure carries a code, a layer
(`rag` or `evaluation`) and a `retryable` flag (EVAL-011). `evaluate_batch_safe`
returns them as `{"error": {...}}` so the API layer maps codes to status codes
without parsing messages.

**Retries are narrow on purpose.** RAG-017 retries only an *unusable response
body* — invalid JSON, a schema violation, or an empty response. Three classes are
excluded: transport failures (the Anthropic SDK already retries those with
backoff, so a second loop would multiply worst-case latency without improving
reliability), refusals (a deterministic policy outcome — re-sending identical
input can only burn budget), and configuration errors such as an inconsistent
rubric (three attempts would just delay the real signal). Every attempt's token
usage is still summed into the audit line, so a retried call is billed honestly.

**Truncation and degradation are always visible.** A truncated prompt sets
`prompt_truncated`, a capped justification lands in `justification_truncated`, a
missing calibration example sets `calibration_used: false`, and a derived batch
summary is labelled `source: "derived"`. Nothing degrades silently.

## Verification

```
pytest tests/member4 -q     ->  187 passed
```

Environment note: the suite was verified in a lightweight virtualenv containing
`pytest`, `pydantic`, `anthropic`, `python-dotenv`, `numpy`, `pandas` and
`chromadb`. `sentence-transformers` was **not** installed, and Member 4's tests
still pass because the embedder is imported lazily and injected in tests. Member
3's tests do require it (11 of their files fail to collect without it) — that is
an environment gap in this sandbox, not a code change.

## Open items handed forward

1. **EVAL-008 / C5 ethics policy — `[DATA NEEDED]`.** Written confirmation from
   Team Beta / Alliance Compliance on whether the hard-gate veto is active has
   not been received. Build runs weighted-only; the gate is implemented and
   tested behind `PECS_ETHICS_HARD_GATE`; every result records the pending status.
2. **RAG-012 / prompt caching.** Implemented correctly, but Claude Haiku 4.5's
   4,096-token minimum cacheable prefix is larger than the condensed rubric, so
   no cache hit will occur. Reported per result in `meta.caching`. Team decision
   needed: lengthen the cached prefix or change model.
3. **Eager ChromaDB client (Member 3's VDB-006).** Importing any vectorstore
   submodule constructs a `PersistentClient` and creates `Database/` on disk.
   `Database/` was added to `.gitignore` (FOUND-003 already required this); a
   separate task proposes making the client lazy. Not a blocker.
