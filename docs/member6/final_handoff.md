# Member 6 → Final Delivery Handoff

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)
Covers the Section 8 checklist for this handoff, Gates 6 and 7 from Section 9,
and the Section 10 Definition of Done.

## Section 8 checklist — Member 6 → Final Delivery

- [x] **Full benchmark report completed against the gold set** —
      `benchmark/benchmark_report.md`. Retrieval quality is measured; the
      grading, consistency, edge-case and latency sections read **NOT MEASURED**
      because no `ANTHROPIC_API_KEY` was available. The report says so on its
      face rather than omitting the sections.
- [x] **Full test suite passing with a published test report** —
      `tests/test_report.md`: **804 passed, 0 failed, 0 skipped**, 83% line
      coverage, generated from pytest's own JUnit and coverage XML.
- [ ] **Docker build verified locally and deployed successfully** — the image,
      compose file, ignore file and verification script exist and their
      configuration invariants are asserted by 36 tests, but **no build has been
      executed** (no Docker engine) and **nothing has been deployed** (no
      confirmed host).
- [x] **Deployment, rollback, and known-issues documentation complete** —
      `docs/deployment.md`, `docs/deployment_config.md`,
      `docs/rollback_procedure.md`, `docs/known_issues_and_limitations.md`.
- [x] **Final Git release tag created** — `v1.0`, local. Pushing it is the
      team's call; see §"What closes the remaining gaps".
- [ ] **Team 2 sign-off obtained from all members** — HAND-009 is a team
      activity and has not been run.

Two boxes are unticked on purpose. Ticking them would be the one thing this
handoff cannot afford: a claim that something was verified when it was not.

## What you are receiving

**Benchmark harness** (`benchmark/`) — a complete, tested measurement path from
the frozen gold baseline to a generated report. Three evaluator adapters (`http`
against a running API, `engine` in-process, `replay` from a recording) behind one
interface, so a benchmark run is reproducible whatever infrastructure exists.
Run it with `python -m benchmark.run_benchmark`.

**Test suite** (`tests/member6/`, 309 tests) — dataset validation, embeddings,
vector store, RAG assembly, scoring, mocked LLM behaviour, end-to-end pipeline,
regression fixtures, malformed-input hardening, and deployment configuration.
Runs offline: no API key, no network, no dependence on the production store.

**Container and deployment assets** — `Dockerfile`, `docker-compose.yml`,
`.dockerignore`, `scripts/verify_container.py`, and the four deployment
documents.

**A working vector store** — `scripts/build_vector_store.py` populates ChromaDB
from Member 3's embedding export (241 train-split records; validation and test
splits deliberately excluded to keep the benchmark honest). This did not exist
before: the documented ingestion path fails on the artifact as shipped — see
issue #4 in the known-issues document.

## Three things worth reading before anything else

**1. There is no grading-accuracy number for this release.**
No MAE, no correlation, no consistency figure. The harness is verified — a
scripted evaluator that echoes the baseline yields MAE 0 and correlation 1, and a
uniformly generous one yields exactly the offset it was given — but a verified
harness is not a measured system. If someone asks how accurate the grader is,
the answer is "not yet measured", and the command that changes that is in
`docs/known_issues_and_limitations.md` §1. One live run costs 8 Haiku calls.

**2. An empty vector store is silent, not loud.**
RAG-006 treats "no calibration available" as a valid state, so a service with an
empty store returns scores — just worse ones. Check
`evaluations[].calibration_used` after every deploy. This is the single most
likely way for the system to be quietly broken in production.

**3. C5 does not veto anything in this build.**
A submission scoring 9/9/9/9/**2**/9 composites to 7.95 and is certified
`strong`. That is the documented, unconfirmed policy state (EVAL-008), recorded
in every response as `ethics_gate.policy_confirmation`. If Alliance Compliance
intends the hard gate to be active, it is one environment variable and a
regression-fixture update away — and the fixture is deliberately arranged so the
change cannot be made silently.

## Verifying this handoff yourself

```bash
pip install -r requirements.txt -r requirements-dev.txt

pytest -q                                        # expect 804 passed
python -m scripts.generate_test_report           # regenerates tests/test_report.md
python -m scripts.build_vector_store             # 241 records ingested
python -m benchmark.run_benchmark --only retrieval --run-id verify
python -m benchmark.report                       # regenerates the benchmark report
python -m scripts.verify_container               # exits 2 without Docker, by design
```

Each command either produces the artifact it names or explains why it cannot.

## What closes the remaining gaps

| Gap | Blocker | Command that closes it |
|---|---|---|
| Grading quality (BENCH-004/005/006) | `ANTHROPIC_API_KEY` | `python -m benchmark.run_benchmark --mode engine --run-id live-001 && python -m benchmark.report` |
| Consistency (BENCH-007) | same | `python -m benchmark.run_benchmark --mode engine --only consistency --trials 5` |
| Edge cases (BENCH-008) | same | `python -m benchmark.run_benchmark --mode engine --only edge` |
| Latency / cost (BENCH-009) | same | included in the grading run above |
| Container verification (DEP-001/003/004/005/006) | Docker engine | `python -m scripts.verify_container` |
| Deployment (DEP-007/008) | DevOps host confirmation | fill in `docs/deployment_config.md` §3, then `docs/deployment.md` §7 |
| Rollback drill (DEP-009) | a deployed host | `docs/rollback_procedure.md` §5 |
| Team sign-off (HAND-009) | the team | run the demo against mock downstream payloads |

## Section 10 — Definition of Done

The tracker states these are completion conditions, not claims. Against this
build:

| Condition | Status |
|---|---|
| Dataset validated; coverage documented | ✅ documented — one domain in the store (issue #2) |
| Embeddings reproducible and dimensionally verified | ✅ `tests/member6/test_embedding.py` |
| ChromaDB populated, persistent, survives restart | ✅ locally · ⛔ not verified in a container |
| Retrieval returns correct, metadata-intact results | ✅ 10/10 self-retrieval top-1 |
| RAG assembles context and builds the request correctly | ✅ `test_rag_context.py`, `test_e2e.py` |
| Evaluator returns schema-valid C1–C6 JSON | ✅ against a scripted model · ⛔ never against a live one |
| Scoring matches the canonical formula, no ×10 error | ✅ hand-calculated tests + fixtures |
| Benchmark against the gold set completed and reported honestly | ⛔ **not completed** — reported honestly as such |
| FastAPI implemented, authenticated, rate-limited, documented | ✅ |
| API integration tests pass, incl. malformed input | ✅ 804 passing |
| Downstream contract documented and accepted | ✅ documented · ⛔ acceptance not recorded |
| Docker build succeeds and runs locally | ⛔ **not executed** |
| Deployment succeeds and is verified live | ⛔ **not executed** |
| All documentation up to date | ✅ |
| No secrets committed | ✅ `.env` absent; `.gitignore` and `.dockerignore` exclude it |
| Repository clean, branches merged or closed | ⚠️ this branch is unmerged |
| Final release tag on `main` | ⚠️ `v1.0` tagged locally, not pushed |
| Downstream teams can consume the API | ⛔ no deployed endpoint to consume |
| Final end-to-end demo succeeds | ⛔ HAND-009 not run |
| All six members signed off | ⛔ not obtained |

**Team 2 cannot declare "Implementation Complete" on this build.** Seven
conditions are unmet, and every one of them needs a credential, an engine, a
host, or a person — not more code.
