# Known Issues and Limitations

Task: **HAND-007** · Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Every limitation below is traceable to a specific test, benchmark run, or
artifact in this repository — that is HAND-007's acceptance criterion, and the
"Evidence" column is where it is discharged. Nothing here is a suspicion.

Severity is about impact on a downstream consumer of `/evaluate`, not about how
hard the fix is.

| # | Issue | Severity | Blocks release? |
|---|---|---|---|
| 1 | Grading quality never measured against a live model | **High** | For a quality claim, yes |
| 2 | Calibration corpus is CODE-only; four master domains unrepresented | **High** | For non-CODE traffic, yes |
| 3 | Gold "reference" scores are judge consensus, not SME scores | **High** | For a correlation claim, yes |
| 4 | EMB-006 export is incompatible with the VDB-003/VDB-004 schemas | **High** | Worked around; fix owed |
| 5 | No deployment performed; hosting target unconfirmed | **High** | For DEP-008, yes |
| 6 | Docker build/health/persistence unverified on this machine | Medium | Verification owed |
| 7 | C5 ethics policy unconfirmed (EVAL-008) | Medium | No — behaviour is recorded |
| 8 | Prompt caching produces no cache hits (RAG-012) | Low | No — cost only |
| 9 | Workstream B modules absent from the repository | Medium | No — data exists |
| 10 | Rate limiting is per-process and in-memory | Medium | No — documented |
| 11 | Calibration neighbours skew to low-scoring records | Medium | No — needs review |
| 12 | Dead configuration keys in `.env.example` | Low | No |

---

## 1. Grading quality has never been measured against a live model

**Evidence:** `benchmark/benchmark_report.md` §2 — **NOT MEASURED**;
`benchmark/results/grading_local001.skipped.json` records the reason
(`ANTHROPIC_API_KEY is not set`).

No MAE, correlation, consistency or edge-case figure exists for this build,
because no credential was available in the environment where the work was done.
The harness is complete and verified — `tests/member6/test_benchmark_harness.py`
proves the comparison, metric and reporting path end to end against a scripted
evaluator, including the known-answer case where a perfect evaluator must score
MAE 0 and correlation 1 — but a verified harness is not a measured system.

**Do not** quote a grading-accuracy figure for this release. There is none.

Unblock with:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
python -m scripts.build_vector_store            # if not already populated
python -m benchmark.run_benchmark --mode engine --run-id live-001
python -m benchmark.report
```

Estimated cost: 39 gold records at batch size 5 is 8 Claude Haiku 4.5 calls.

## 2. The calibration corpus covers one domain

**Evidence:** `benchmark/results/retrieval_quality_local001.json` — the store
holds 241 records, 231 `CODE` and 10 `EDGE`. `data/reports/quality_summary.json`
lists six domains in the master dataset and records
`domains_missing_3judge_coverage: [CAREER, CODE, EDGE, GEN]`.
`tests/member6/test_dataset_validation.py::test_metadata_enum_covers_only_the_two_ingested_domains`
pins the VDB-004 enum at `{CODE, EDGE}`.

Two consequences, and the second is the one that bites:

- A `CAREER`, `CREATE`, `EDU` or `GEN` submission cannot be domain-scoped —
  VDB-004 rejects those values outright.
- An *unscoped* submission in those domains still gets graded, using whatever
  CODE record happens to be nearest. `RQ-NEAR-003` in the retrieval artifact
  shows this concretely: a creative-writing probe retrieved a CODE record at
  distance 1.51. The result is not an error; it is a quietly worse score.

The gold calibration seed (`data/processed/gold_calibration_seed.json`) contains
the CREATE and EDU examples ANN-006 curated, but they carry
`review_status: candidate_needs_sme_signoff` and are not in the store.

## 3. The gold baseline is judge consensus, not human reference scores

**Evidence:** `benchmark/gold_baseline.json` → `reference_provenance`; all 39
records carry `score_source: auto_mean`. `data/reports/quality_summary.json`
records `human_review_status: pending_sme_signoff`.

The frozen baseline is the mean of three LLM judges (ChatGPT, Gemini, Claude),
not scores assigned by a subject-matter expert. Any correlation computed against
it measures **agreement with a judge ensemble**, not agreement with human
graders — and the programme's >0.85 target is stated in terms of human
agreement.

`benchmark/correlation.py` refuses to interpret a coefficient below a 30-pair
sample and labels the result `insufficient_sample`; at 39 records the sample is
marginal even once scores exist. **[DATA NEEDED]** — minimum gold-set size
sign-off from Team Beta.

## 4. The embedding export cannot be ingested by the vector store

**Evidence:** `scripts/build_vector_store.py` module docstring;
`bulk_ingest_from_serialized` raises
`ValueError: Missing required metadata keys: ['c1', …, 'level', 'source', 'version']`
on `data/processed/embeddings.json` as shipped.

The handoff path documented in `docs/embedding.md` §12
(`collection.upsert(**load_serialized(path))`) does not work on the artifact
EMB-006 actually produces:

| | EMB-006 export | VDB-003/VDB-004 requires |
|---|---|---|
| Metadata keys | 32, including `prompt_id`, `skill_level`, `c1_score`… | exactly 10: `domain`, `level`, `c1`…`c6`, `source`, `version` |
| Document | raw prompt text | JSON `{"scores": {...}, "justification": "..."}` |

The second row matters more than it looks: raw text ingests fine into ChromaDB
and retrieves fine, but RAG-008's formatter cannot parse it, so
`build_examples()` skips the record and every evaluation silently runs
uncalibrated.

**Worked around, not fixed.** `scripts/build_vector_store.py` adapts the export
losslessly and ingests through the official VDB-012 → VDB-007 path. The fix
belongs in `src/embeddings/embedding_serializer.py` (Member 3), after which the
adapter should be deleted.

Also missing: VDB-012's `docs/vectorstore_hardening.md`. The task ID was
reassigned in-repo to `src/vectorstore/bulk_ingest.py`, so no backup or
persistence-hardening note exists — which is why
`docs/rollback_procedure.md` §4 names rebuild-from-source as the primary
recovery route rather than restore-from-backup.

## 5. Nothing has been deployed

**Evidence:** `docs/deployment_config.md` §3 — every host-specific value is
`[DATA NEEDED]`. `docs/rollback_procedure.md` §5 — the drill table reads
_not yet run_.

DEP-007 requires the hosting target and credentials owner to be confirmed with
the DevOps Team; that confirmation has not arrived, so DEP-008 (deploy and
verify the deployed endpoints) and DEP-009's rollback drill are both blocked.
The build, run, verify and rollback procedures are written and host-agnostic
(`docs/deployment.md`); what is missing is a host to run them against.

## 6. The container has not been built or run on this machine

**Evidence:** `benchmark/results/container_verification.json` —
`{"status": "docker_unavailable"}`.

No Docker engine was available, so these acceptance criteria are **written and
scripted but not executed**: DEP-001 (`docker build` completes), DEP-004
(container health check answers), DEP-003 (data survives a restart), DEP-005 (no
model re-download on second start), DEP-006 (`docker history` shows no secret).

What *is* verified is every property decidable from the artifacts:
`tests/member6/test_deployment.py` (36 tests) checks the multi-stage build, the
exposed port, the non-root user, the health check, the pre-baked model matching
`MODEL_NAME` in the source, the store path agreeing with the volume mount, and
`.dockerignore` excluding `.env`.

Run `python -m scripts.verify_container` on a machine with Docker. It exits 2
and records `docker_unavailable` rather than reporting a green result it did not
earn.

## 7. C5 ethics handling is unconfirmed policy

**Evidence:** `tests/member6/fixtures/regression_cases.json` → `REG-SCORE-007`
records `ethics_mode: weighted_only`; every API response carries
`ethics_gate.policy_confirmation: pending_alliance_compliance_confirmation`.

Inherited from Member 4 and unchanged. C5 contributes its 15% weight and vetoes
nothing. A submission scoring 9/9/9/9/**2**/9 composites to 7.95 and is reported
`strong` — a prompt with a serious ethics failure certified as strong.

The hard-gate path is implemented and tested behind `PECS_ETHICS_HARD_GATE`.
**[DATA NEEDED]** — written confirmation from Team Beta / Alliance Compliance.
Enabling it changes stored scores, which is why `REG-SCORE-007` is fixtured: the
switch cannot be flipped without a failing test drawing attention to it.

## 8. Prompt caching produces no cache hits

**Evidence:** `meta.caching` on every evaluation result;
`benchmark/perf.py::measure_run` reports `cache_hits_observed`.

RAG-012 marks the rubric block for ephemeral caching correctly, but Claude Haiku
4.5 requires a 4,096-token minimum cacheable prefix and the condensed rubric is
well below it, so the API declines to cache. `cache_read_input_tokens` is always
0. Cost impact only; the fix (lengthen the prefix, or change model) is a team
decision with its own cost implications.

## 9. Workstream B modules are not in the repository

**Evidence:** `src/dataset/` contains only `annotation/`. None of DS-002's
`schema.py`, DS-003's `validate_schema.py`, DS-004's `enums.py`, DS-008's
`quarantine.py`, DS-013's `dataset_stats.py` or DS-014's `split_dataset.py`
exists. `docs/` has no `dataset_inventory.md` or `dataset_quality_report.md`.

The *outputs* of that work are present — `data/processed/`, `data/reports/`, the
splits, the leakage report — so the dataset itself is validated and the pipeline
runs. What is missing is the reusable, testable code, which is why TEST-001
covers the annotation-layer and vector-store-layer validation that exists
instead. Re-running Member 1's cleaning on a new corpus is currently not
reproducible from this repository.

## 10. Rate limiting is per-process and resets on restart

**Evidence:** `src/api/rate_limit.py` — an in-memory `deque` on a module-level
singleton.

30 requests/minute **per process**. Two replicas allow 60/minute; a restart
clears the window. It is a runaway-client safety valve, not a quota. A real
global limit belongs at the ingress — noted in `docs/deployment_config.md` §4.

## 11. Retrieved calibration examples skew toward low-scoring records

**Evidence:** `benchmark/results/retrieval_quality_local001.json` — all three
near-pair probes retrieved `weak`-band records as their top match, at distances
0.70–1.51, despite the probes themselves being well-formed prompts.

The ingested corpus (train split, 241 records) is 109 developing / 101 weak /
31 strong, so only 13% of the calibration examples available to retrieval are
strong ones, and the nearest neighbour of an arbitrary prompt is far more likely
to be a low-scoring example than a high-scoring one. Whether an evaluator anchors on a weak calibration example is exactly
the kind of effect the benchmark exists to measure, and it has not been measured
(see #1). Flagged here so the question is asked when the live run happens:
compare `evaluate_batch(..., retrieve_calibration=False)` against the calibrated
run, which the harness supports directly.

## 12. Dead configuration keys

**Evidence:** `CHROMA_PERSIST_DIRECTORY` and `CHROMA_COLLECTION_NAME` appear in
`.env.example` but are read nowhere — the live settings are `CHROMA_DB_PATH`
(VDB-001) and the hard-coded `COLLECTION_NAME` (VDB-002).

Harmless, but misleading: setting `CHROMA_PERSIST_DIRECTORY` and expecting the
store to move is a plausible mistake. Annotated in place rather than deleted, so
the removal is a deliberate change by whoever owns FOUND-003.

---

## What Member 6 changed in previous milestones

Three defects were found by running the inherited suite before writing new code
(Section 7, step 2 of the tracker). All three are fixed on this branch; each is
recorded here because they were other members' modules.

| Defect | Symptom | Fix |
|---|---|---|
| `ingest_record` bypassed `upsert_record` | VDB-005 wrote through `collection.upsert` directly, so VDB-007's duplicate-ID semantics were bypassed. `test_ingest_delegates_to_upsert` failed in isolation. | `src/vectorstore/ingest.py` now delegates to VDB-007. Regression-guarded by `tests/member6/test_vectorstore.py::test_ingest_record_writes_through_the_upsert_layer`. |
| Ephemeral ChromaDB clients share one database | `chromadb.EphemeralClient()` instances with identical settings share an in-process system, so `test_retrieval.py` and `test_filtered_retrieve.py` overwrote each other's `record-A/B/C` and 9 tests failed whichever ran second. Both files passed alone. | Both fixtures now drop the collection before seeding and tear it down after. Member 6's own store tests use per-test collection names. |
| `Database/` created at import time | `src/vectorstore/client.py` built a `PersistentClient` at module import, so importing the package created the directory — making `test_config_does_not_initialize_chromadb` order-dependent, and letting a bare `python -c "import src.api.main"` create a stray unmounted `Database/` inside the container. | Client initialization deferred to first `get_client()` call; the singleton guarantee is unchanged. The config test now asserts "resolution creates nothing" rather than "no store exists anywhere". |

Suite before: 482 passed, 13 failed. After: **804 passed, 0 failed**
(`tests/test_report.md`).
