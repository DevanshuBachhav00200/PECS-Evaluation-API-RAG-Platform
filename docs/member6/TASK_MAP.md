# Member 6 — Task ID → Deliverable Map

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment) — Workstreams H,
K, L and part of M.

Every task from Sections 4/5 of `PECS_Team2_Implementation_Task_Tracker_v1.0`
with the file that implements it, the test or artifact that proves its
acceptance criterion, and its honest status.

Run the suite:

```bash
pytest tests/member6 -q          # 309 tests
pytest -q                        # 804 tests, whole repository
```

Status key — **done** (acceptance criterion met and evidenced) ·
**blocked** (implemented, criterion needs an unavailable precondition) ·
**partial** (criterion met for what exists; a gap is recorded).

---

## Workstream H — Benchmarking

| Task ID | Deliverable in this repo | Evidence | Status |
|---|---|---|---|
| BENCH-001 | `benchmark/test_set_loader.py` | `test_benchmark_harness.py::test_gold_eval_set_loads_with_zero_schema_errors` (39 records, real CSV) | **done** |
| BENCH-002 | `benchmark/gold_baseline.py` → `benchmark/gold_baseline.json` | `::test_baseline_id_is_stable_across_regeneration`, `::test_tampering_with_the_baseline_file_is_detected`, `::test_every_run_records_the_baseline_it_was_scored_against` | **done** |
| BENCH-003 | `benchmark/retrieval_quality.py` | `benchmark/results/retrieval_quality_local001.json` — 10/10 self-retrieval top-1, 2/2 domain scoping, 3 near-pair probes logged for review | **done** |
| BENCH-004 | `benchmark/grading_comparison.py`, `benchmark/evaluators.py` | `::test_comparison_table_has_one_row_per_record_per_criterion`, `::test_a_perfect_evaluator_scores_zero_error` | **blocked** — harness verified, no live run (no API key) |
| BENCH-005 | `benchmark/mae.py` | `test_benchmark_metrics.py::test_mae_matches_a_hand_calculation` (+ 12 more) | **done** |
| BENCH-006 | `benchmark/correlation.py` | `::test_pearson_matches_a_hand_calculation`, `::test_small_sample_is_computed_but_flagged_uninterpretable` | **done** — result uninterpretable pending sample-size sign-off |
| BENCH-007 | `benchmark/consistency_test.py` | `test_benchmark_stages.py::test_the_batch_is_identical_on_every_trial`, `::test_fewer_than_five_trials_is_rejected` | **blocked** — needs 5 live trials |
| BENCH-008 | `benchmark/edge_cases.py` | `::test_every_required_edge_case_category_is_covered`; malformed cases asserted in `test_malformed_input.py` | **partial** — malformed category run; the rest need a live model |
| BENCH-009 | `benchmark/perf.py` | `::test_performance_summary_is_derived_from_the_run_artifact`, `::test_scripted_run_is_not_valid_for_the_sla` | **blocked** — no live run to measure |
| BENCH-010 | `benchmark/report.py` → `benchmark/benchmark_report.md` | `::test_report_marks_absent_sections_as_not_measured`, `::test_report_flags_a_replayed_run` | **done** — report generated; grading sections read NOT MEASURED |

Runner: `benchmark/run_benchmark.py` (`python -m benchmark.run_benchmark`).

## Workstream K — Testing

| Task ID | Deliverable in this repo | Tests | Status |
|---|---|---|---|
| TEST-001 | `tests/member6/test_dataset_validation.py` | 36 | **partial** — Workstream B modules absent; covers the validation that exists |
| TEST-002 | `tests/member6/test_embedding.py` | 13 | **done** |
| TEST-003 | `tests/member6/test_vectorstore.py` | 19 | **done** |
| TEST-004 | `tests/member6/test_rag_context.py` | 18 | **done** |
| TEST-005 | `tests/member6/test_scoring_and_bands.py` | 31 | **done** |
| TEST-006 | `tests/member6/test_llm_mock.py` | 21 | **done** |
| TEST-007 | `tests/member6/test_e2e.py` | 8 | **done** |
| TEST-008 | `tests/member6/test_regression.py` + `tests/member6/fixtures/regression_cases.json` | 7 | **done** |
| TEST-009 | `tests/member6/test_malformed_input.py` | 32 | **done** |
| TEST-010 | `scripts/generate_test_report.py` → `tests/test_report.md` | — | **done** — 804 passed, 0 failed, 83% line coverage |

Supporting suites (not numbered tasks, required to trust the above):

| Component | File | Tests |
|---|---|---|
| Benchmark metrics | `tests/member6/test_benchmark_metrics.py` | 28 |
| Benchmark harness | `tests/member6/test_benchmark_harness.py` | 41 |
| Benchmark stages + runner | `tests/member6/test_benchmark_stages.py` | 19 |
| Deployment configuration | `tests/member6/test_deployment.py` | 36 |
| Reusable fakes | `tests/member6/m6_fakes.py`, `tests/member6/conftest.py` | — |

> **Filename note.** The tracker lists TEST-005's deliverable as
> `tests/test_scoring.py`. It is `tests/member6/test_scoring_and_bands.py` here:
> the repository keeps each member's suite in its own directory, those
> directories are not packages, and `tests/member2/test_scoring.py` already
> exists — two modules with the same basename break pytest collection outright.

## Workstream L — Deployment

| Task ID | Deliverable in this repo | Evidence | Status |
|---|---|---|---|
| DEP-001 | `Dockerfile` (multi-stage, port 9000) | `test_deployment.py` — multi-stage, EXPOSE 9000, non-root, health check, no tests/data copied | **blocked** — `docker build` not executed (no engine) |
| DEP-002 | `.env.example` + `docker-compose.yml` `env_file` | `::test_required_variable_is_declared_in_env_example`, `::test_env_example_ships_no_real_secret` | **done** |
| DEP-003 | `docker-compose.yml` volume mount | `::test_compose_mounts_the_vector_store_as_a_volume`, `::test_store_path_and_mount_point_agree`; runtime check in `scripts/verify_container.py` | **blocked** — restart check needs Docker |
| DEP-004 | `scripts/verify_container.py` | `benchmark/results/container_verification.json` → `docker_unavailable` | **blocked** — no Docker engine |
| DEP-005 | Dockerfile model pre-bake | `::test_prebaked_model_matches_the_one_the_code_loads` (reads `MODEL_NAME` from source) | **blocked** — no-redownload check needs Docker |
| DEP-006 | `.dockerignore` | `::test_dockerignore_excludes_secret_material`, `::test_settings_repr_does_not_leak_secret_values` | **partial** — `docker history` check needs Docker |
| DEP-007 | `docs/deployment_config.md` | every host value marked `[DATA NEEDED]` | **blocked** — DevOps confirmation |
| DEP-008 | — | — | **blocked** — depends on DEP-007 |
| DEP-009 | `docs/rollback_procedure.md` | drill table records _not yet run_ | **blocked** — needs a deployed host |
| DEP-010 | `docs/deployment.md` | full build → run → verify → deploy → rollback procedure | **done** |

Supporting: `scripts/build_vector_store.py` — populates the store the container
mounts, and bridges the EMB-006 ↔ VDB-004 schema gap (issue #4 in
`docs/known_issues_and_limitations.md`).

## Workstream M — Handoff

| Task ID | Deliverable in this repo | Status |
|---|---|---|
| HAND-007 | `docs/known_issues_and_limitations.md` | **done** — 12 issues, each with evidence |
| HAND-008 | `CHANGELOG.md` + git tag `v1.0` | **partial** — tag created locally; several DoD items in Section 10 are not satisfied (see below) |

---

## What is not green, and why

Three preconditions were unavailable in the environment this work was done in.
None is a coding gap; all three are recorded rather than worked around.

1. **No `ANTHROPIC_API_KEY`** → BENCH-004/007/009 and most of BENCH-008 could
   not be run. The harness is verified against a scripted evaluator, including
   the known-answer case that a perfect evaluator scores MAE 0 and correlation 1.
2. **No Docker engine** → DEP-001/003/004/005 and part of DEP-006 could not be
   executed. Every property decidable from the artifacts is asserted by
   `tests/member6/test_deployment.py`; the rest is scripted in
   `scripts/verify_container.py`, which reports `docker_unavailable` rather than
   a result it did not earn.
3. **No confirmed hosting target** (`[DATA NEEDED]`, DevOps) → DEP-007/008 and
   DEP-009's drill are blocked.

`docs/known_issues_and_limitations.md` carries the full list with evidence and
the exact commands that close each gap.
