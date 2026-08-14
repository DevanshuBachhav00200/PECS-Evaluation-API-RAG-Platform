# Changelog

All notable changes to the PECS Evaluation RAG API project will be documented here.

## [v1.0] — Team 2 release candidate

Tagged by **HAND-008** (Member 6). This tag marks the point at which every
Team 2 workstream has been implemented and the whole suite is green. It does
**not** assert that Section 10's Definition of Done is satisfied — seven
conditions are not, and `docs/member6/final_handoff.md` lists them individually.
Read `docs/known_issues_and_limitations.md` before quoting anything about this
build's grading quality.

### Added — Member 6, Workstream H (benchmarking, BENCH-001…010)
- `benchmark/test_set_loader.py` — loads Member 2's held-out gold evaluation set
  into benchmark-ready records, rejecting duplicate ids, out-of-range scores,
  off-scale composites (the ×10 error) and split leakage.
- `benchmark/gold_baseline.py` + `benchmark/gold_baseline.json` — the frozen
  comparison baseline, fingerprinted with SHA-256 and integrity-checked on every
  load, referenced by `baseline_id` on every run artifact.
- `benchmark/evaluators.py` — three adapters behind one interface: `http`
  (against a running `/evaluate`), `engine` (in-process), `replay` (offline,
  from a recording). Every run records which produced it.
- `benchmark/grading_comparison.py` — the per-record, per-criterion comparison
  harness. Failed batches keep their rows so a failure cannot flatter the MAE.
- `benchmark/mae.py`, `benchmark/correlation.py` — MAE, max error, bias,
  tolerance rate, Pearson and Spearman, in dependency-free Python so each is
  verifiable by hand. Correlation refuses to be interpreted below a sample-size
  threshold and returns `meets_target: None` rather than a verdict.
- `benchmark/consistency_test.py` — repeated-trial variance with batch
  composition held constant and a 5-trial floor enforced.
- `benchmark/edge_cases.py` — 11 cases across the five required categories, each
  declaring the behaviour it expects and why.
- `benchmark/perf.py` — latency, token and cost roll-ups per batch size, derived
  entirely from stored run artifacts.
- `benchmark/report.py` + `benchmark/benchmark_report.md` — the generated report.
  Sections with no artifact print **NOT MEASURED** and the command that produces
  them; replayed runs render a warning banner.
- `benchmark/run_benchmark.py` — one CLI driving BENCH-003…010. A stage that
  cannot run records why and the run continues.

### Added — Member 6, Workstream K (testing, TEST-001…010)
- `tests/member6/` — 309 tests: dataset validation, embedding dimensionality and
  determinism, vector-store upsert/retrieval/filtering, RAG context assembly,
  scoring and band boundaries, mocked LLM parsing and retries, an end-to-end
  pipeline test through the real stack, regression fixtures, malformed-input
  hardening against the live endpoint, benchmark-harness self-tests, and
  deployment-configuration invariants.
- `tests/member6/fixtures/regression_cases.json` — known-good outputs of the
  deterministic evaluation layers, diffed on every run.
- `scripts/generate_test_report.py` + `tests/test_report.md` — the test report,
  generated from pytest's own JUnit and coverage XML rather than transcribed.
- `requirements-dev.txt` — test-only dependencies, kept out of the runtime image.

### Added — Member 6, Workstream L (deployment, DEP-001…010)
- `Dockerfile` — multi-stage, port 9000, non-root, health check, embedding-model
  cache pre-baked so no container downloads it on cold start.
- `docker-compose.yml` — bind-mounted `Database/` so the vector store survives
  restarts and can be extended without a rebuild.
- `.dockerignore` — keeps `.env`, keys, tests, data and the git history out of
  the build context entirely.
- `scripts/verify_container.py` — executable verification of DEP-001/003/004/005/006.
  Reports `docker_unavailable` and exits 2 when no engine is present rather than
  emitting a result it did not earn.
- `docs/deployment.md`, `docs/deployment_config.md`, `docs/rollback_procedure.md`.

### Added — Member 6, Workstream M and supporting work
- `docs/known_issues_and_limitations.md` (HAND-007) — 12 issues, each traced to a
  test, artifact or run.
- `docs/member6/` — task map, implementation notes, final handoff.
- `scripts/build_vector_store.py` — populates ChromaDB from Member 3's embedding
  export, bridging the EMB-006 ↔ VDB-003/VDB-004 schema gap. Excludes the
  validation and test splits by default so the benchmark cannot retrieve its own
  gold answers.

### Fixed — defects found in earlier milestones
- **VDB-005:** `ingest_record` wrote through `collection.upsert` directly,
  bypassing VDB-007's duplicate-ID semantics. It now delegates to
  `upsert_record`, and a regression test pins the delegation.
- **VDB-006:** `src/vectorstore/client.py` constructed its `PersistentClient` at
  module import, so importing the package created `Database/` as a side effect —
  making a test order-dependent and letting the container's health-check path
  create a stray unmounted directory inside the image. Initialization is now
  deferred to the first `get_client()` call; the singleton guarantee is unchanged.
- **tests/member3:** `chromadb.EphemeralClient()` instances with identical
  settings share one in-process database, so `test_retrieval.py` and
  `test_filtered_retrieve.py` overwrote each other's fixture records and nine
  tests failed whichever ran second. Both fixtures now drop and recreate their
  collection.
- **tests/member3:** `test_config_does_not_initialize_chromadb` asserted that no
  `Database/` directory exists anywhere, which fails for anyone who has actually
  built the store. It now asserts the property VDB-001 promises — that path
  resolution creates nothing.

Suite before this branch: 482 passed, 13 failed. After: **804 passed, 0 failed.**

### Known at release
- **No live grading measurement.** BENCH-004/005/006/007/009 are implemented and
  self-tested but were never run against the model — no `ANTHROPIC_API_KEY` was
  available. There is no MAE or correlation figure for this build.
- **No container build and no deployment.** No Docker engine was available, and
  the hosting target is unconfirmed (DEP-007, `[DATA NEEDED]`, DevOps).
- **Calibration corpus is CODE-only** (231 CODE / 10 EDGE of 241 ingested);
  four of the six master domains have no coverage.
- **Gold reference scores are 3-judge consensus means**, not SME scores, pending
  Team Beta sign-off — so any future correlation figure measures agreement with a
  judge ensemble, not with humans.

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