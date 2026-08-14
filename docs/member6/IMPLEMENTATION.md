# Member 6 — Implementation Notes

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Why the Workstream H/K/L code is shaped the way it is. The task-by-task map is in
`TASK_MAP.md`; this is the reasoning behind the non-obvious choices, so the next
person changing this code knows which parts are load-bearing.

---

## The benchmark harness

### Three adapters behind one interface

BENCH-004 says "run the gold test set through `/evaluate`". Three things can
legitimately mean that, and `benchmark/evaluators.py` supports all three:

| Mode | Path | Needs | Used for |
|---|---|---|---|
| `http` | POST to a running `/evaluate` | live service + key | the literal criterion; also exercises Member 5's auth and error mapping |
| `engine` | `evaluate_batch()` in-process | key + populated store | same evaluation code, one hop shorter |
| `replay` | recorded outputs, keyed by `record_id` | nothing | CI, and re-deriving a published report months later |

The alternative — hard-wiring the HTTP path — would have made the metric layer
untestable without a paid API call, which in practice means untested.

Every run artifact records its `mode`, and `benchmark/report.py` renders a
warning banner for any replayed run. A replayed number can never be presented as
a live measurement, even by accident.

### Record ids travel alongside the payload, not inside it

`grade(qa_pairs, record_ids)` takes two parallel sequences rather than a payload
carrying a `_record_id` key. Slightly more awkward; the reason is that the
graded request must be byte-identical to a production one. A benchmark-only key
in the body would be a difference between what is measured and what is served —
small, invisible, and exactly the sort of thing that invalidates a comparison
quietly. `test_submissions_are_sent_without_benchmark_only_keys` pins it.

### The baseline verifies itself

`benchmark/gold_baseline.json` carries a SHA-256 of its own records, recomputed
on every load. Editing a reference score fails the load rather than silently
rebasing every historical comparison. The fingerprint deliberately covers only
the reference data, not the generation metadata, so regenerating from an
unchanged gold set reproduces the same `baseline_id` byte for byte — otherwise
"version-locked" would mean nothing.

### A failed batch keeps its rows

When an evaluation call fails, the records in that batch stay in the comparison
table with null AI scores and the error code. Dropping them would shrink the
denominator, and a shrinking denominator makes MAE look *better* the more often
the system fails. `test_a_failed_batch_is_recorded_not_dropped` guards it.

### Metrics are plain Python

No numpy in `mae.py` or `correlation.py`. BENCH-005's acceptance criterion is
verification against a hand calculation, and a five-line formula anyone can check
on paper is worth more here than a vectorised one nobody audits. It also means
the metric layer imports nothing and runs anywhere.

Two design details worth keeping:

- **Bias is reported next to MAE.** MAE alone cannot distinguish "scatters both
  ways" from "grades every submission a point high" — different defects with
  different fixes.
- **Pearson returns `None`, not 0, for a constant series.** An evaluator that
  returns the same composite for everything is broken; reporting `r = 0` would
  file that under "uncorrelated".

### Correlation refuses to be interpreted early

BENCH-006 is marked *[Requires Validation]* in the tracker. `correlate()` always
computes the coefficient but returns `interpretable: False` and
`sample_size_status: insufficient_sample` below the threshold, and
`meets_target: None` rather than a verdict. The threshold itself is a placeholder
labelled `[DATA NEEDED]` — Team Beta owns the real one.

---

## The test suite

### What is faked, and what deliberately is not

TEST-007 is an end-to-end test, so exactly one thing is substituted: the network
hop to Anthropic, installed at `llm_call._client`. Request assembly, cache
markers, fence stripping, schema validation, retries, the score guard, the
composite, the ethics gate and the response model all execute for real. The
vector store is a real in-memory ChromaDB collection seeded with one correctly
shaped record — not a mock — so retrieval is genuinely exercised without
depending on whatever is in the developer's `Database/`.

The inherited API tests mock `evaluate_batch_safe` at the endpoint, which is
right for testing the endpoint. It would be wrong here: it would test the mock.

### Offline is enforced, not requested

`test_llm_mock.py` monkeypatches `llm_call.get_client` to raise. Any code path
that reaches for a real client fails the test loudly. Without that, the suite
would make paid calls on a developer's laptop and skip in CI — the worst of both.

### A deliberately broken formatter

`test_rag_context.py::test_a_broken_formatter_is_caught_by_the_fixture` swaps in
a formatter that drops the score block and asserts the assembled batch stops
matching. Without it, a silently broken RAG-008 would leave every other test in
that module green: the batch would still have the right number of entries in the
right order, just with no calibration signal in them.

### Regression fixtures cover deterministic layers only

Pinning a sampled model's output as "known good" would fail on ordinary variation
and train everyone to regenerate the fixture without reading the diff. So the
fixtures cover the composite formula, band mapping, ethics block, calibration
formatting and batch assembly — all pure functions. Model stability is BENCH-007's
job, measured as variance rather than asserted as equality.

`REG-SCORE-007` (9/9/9/9/**2**/9 → 7.95, `strong`) exists so that switching the
C5 hard gate on cannot happen silently: the fixture changes, and the suite says so.

### Malformed-input assertions are properties, not codes

`test_malformed_input.py` applies two assertions to every case: the status is one
the integration contract documents, and the body contains no traceback, file
path, or rendered exception class. Asserting only the status code would pass on a
500 that leaks a stack trace — which is the actual risk TEST-009 exists to close.

The malformed cases themselves are imported from `benchmark/edge_cases.py`, so
BENCH-008 and TEST-009 cannot disagree about what those cases are.

---

## Deployment

### The Dockerfile pre-bakes the model

DEP-005. Without it every fresh container downloads ~90 MB before serving its
first request, and a container with no outbound network never becomes healthy at
all. `test_prebaked_model_matches_the_one_the_code_loads` reads `MODEL_NAME` from
`src/embeddings/embedder.py` rather than duplicating the string, so changing the
model without updating the Dockerfile fails a test instead of causing a silent
download in production.

### One worker, scale with replicas

Each uvicorn worker loads its own copy of the embedding model (~1.5 GB resident).
Four workers is four copies for no throughput gain on a synchronous, API-bound
workload.

### `verify_container.py` reports absence

Without a Docker engine it writes `docker_unavailable` per check and exits 2 — a
distinct code from "ran and failed" (1) and "ran and passed" (0). It does not
simulate the checks. A deployment verification that quietly passes when it did
not run is worse than no verification, because someone will cite it.

### The static deployment tests exist because of that

36 tests assert what *is* decidable from the artifacts: multi-stage build, port,
non-root user, health check without credentials, store path matching the volume
mount, `.dockerignore` excluding `.env`, `Settings.__repr__` not leaking secrets.
This is not a substitute for the build; it is the part that can regress between
builds and would otherwise go unnoticed until someone next had Docker to hand.

---

## `scripts/build_vector_store.py`

Not a numbered task. It exists because BENCH-003 needs a populated store,
DEP-003 needs something to populate the mounted volume, and the documented
ingestion path does not work on the artifact EMB-006 actually produces (issue #4
in `docs/known_issues_and_limitations.md`).

Two decisions in it are load-bearing:

**Validation and test splits are excluded by default.** The export contains all
331 records including the 39 the benchmark uses as its gold baseline. Ingesting
those would hand the evaluator each graded record's own reference scores as its
"calibration example", and every MAE figure afterwards would be measuring recall
of a leaked answer. `--include-splits` overrides it and prints a warning to
stderr.

**A record that cannot be represented is rejected, not patched.** The adapter
never guesses a missing domain or invents a justification; it reports the record
and the reason. A silently repaired corpus is a corpus nobody can reason about.

---

## Fixes to earlier milestones

Three defects surfaced by running the inherited suite before writing new code
(Section 7, step 2). Full detail in `docs/known_issues_and_limitations.md`:
`ingest_record` bypassing VDB-007, ephemeral ChromaDB clients sharing one
database across test modules, and `Database/` being created at package import.
Suite before: 482 passed / 13 failed. After: 804 passed / 0 failed.

The third fix is worth a note beyond the test it repaired: an import-time
`PersistentClient` meant that any process importing `src.api.main` — including
the container's health-check path — created a stray, unmounted `Database/` inside
the image layer. Deferring initialization to the first `get_client()` call keeps
the singleton guarantee and removes the side effect.
