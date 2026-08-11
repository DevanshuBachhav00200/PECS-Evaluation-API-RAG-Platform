# Changelog

All notable changes to the PECS Evaluation RAG API project will be documented here.

## [Unreleased]

### Added
- Repository foundation
- Initial project structure
- Development configuration
- **Member 4 — RAG pipeline (RAG-001…RAG-020):** query-time normalization and
  embedding, `retrieve_context()` with configurable Top-K and domain/level
  filtering, structured empty-retrieval handling, compact calibration formatting,
  `build_batch_content()` with per-submission truncation, the `PECS_SYSTEM_PROMPT`
  rubric prompt with ephemeral prompt caching, a single batched Claude call,
  fence-stripped response parsing, C1–C6 schema validation, bounded retries for
  malformed responses, classified API-failure handling, a pre-flight token safety
  abort, and per-call token/cost audit logging.
- **Member 4 — AI evaluation engine (EVAL-001…EVAL-014):** evaluator input schema,
  single-source versioned rubric loading, rubric version + content fingerprint
  tagging, 1–10 score-range enforcement, per-criterion justification word caps,
  the weighted composite score (15/20/20/20/15/10, no ×10 multiplication),
  skill-level band mapping, configurable C5 ethics handling, batch-level
  strengths/gaps/improvement summary, optional confidence parsing, layered
  structured error types, model-version tracking, and the composed
  `evaluate_batch()` / `evaluate_batch_safe()` engine entry point.
- **Member 4 — documentation:** `docs/evaluation_architecture.md`,
  `docs/evaluator_determinism.md`, `docs/member4/` (implementation notes, task
  map, Member 5 handoff).
- **Member 4 — tests:** `tests/member4/` (176 tests) with reusable fakes; runs
  with no ChromaDB, embedding model, or API key.

### Changed
- **Repo-wide (FOUND-006):** added `pytest.ini` with `pythonpath = .` so
  `import src.…` resolves under a bare `pytest` invocation, not only under
  `python -m pytest`. Previously every test module failed collection with
  `ModuleNotFoundError: No module named 'src'` unless run as a module. Also sets
  `testpaths = tests` so an argument-less `pytest` collects only the test suite.
  Requires pytest >= 7.0 (project runs 9.x).
- `.gitignore` now excludes `Database/` (the ChromaDB cache directory called for
  by FOUND-003), so a local vector store cannot be committed.
- **RAG-017:** retry scope narrowed to unusable response bodies only (invalid
  JSON, schema violation, empty response — `RETRYABLE_RESPONSE_CODES`). A
  refusal and a configuration error now surface on the first occurrence instead
  of consuming the whole retry budget.
- **RAG-018:** an empty evaluator response is classified `RAG_LLM_EMPTY_RESPONSE`
  instead of the generic `RAG_LLM_UNKNOWN_ERROR`, which is what makes it
  distinguishable from a refusal for retry purposes.
- **EVAL-003/EVAL-013:** `model_version`, `rubric_version` and `rubric_tag` are
  now stamped on every per-submission evaluation record as well as on the batch
  result, so a stored benchmark row is self-describing.

### Known issues
- **EVAL-008:** C5 is weighted-only pending written confirmation from Team Beta /
  Alliance Compliance on the hard-gate veto. The gate is implemented and tested
  behind `PECS_ETHICS_HARD_GATE`; results record the pending status.
- **RAG-012:** prompt caching is requested correctly but produces no cache hits
  on Claude Haiku 4.5, whose minimum cacheable prefix (4,096 tokens) exceeds the
  condensed rubric. Reported per result in `meta.caching`.

## Release History

### v0.1-repository-foundation
- Initial repository setup
- Project directory structure
- Development configuration