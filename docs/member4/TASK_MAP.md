# Member 4 — Task ID → Deliverable Map

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine) — Workstreams F and G.
Every task from Section 4/5 of `PECS_Team2_Implementation_Task_Tracker_v1.0` with
the file that implements it and the test that proves its acceptance criterion.

Run the suite with:

```bash
pytest tests/member4 -q
```

## Workstream F — RAG Pipeline

| Task ID | Deliverable in this repo | Test |
|---|---|---|
| RAG-001 | `src/rag/query_preprocess.py` | `test_retrieval_context.py::test_query_preprocess_matches_ingestion` |
| RAG-002 | `src/rag/query_embed.py` | `test_retrieval_context.py::test_embed_query_uses_the_shared_emb001_instance`, `::test_embed_query_does_not_reload_the_model` |
| RAG-003 | `src/rag/retrieve_context.py` | `test_retrieval_context.py::test_retrieve_context_returns_one_formatted_example` |
| RAG-004 | `src/rag/config.py` (`DEFAULT_TOP_K`) | `test_retrieval_context.py::test_retrieve_context_top_k_override_changes_request` |
| RAG-005 | `src/rag/filtered_context.py` | `test_retrieval_context.py::test_filtered_context_never_returns_out_of_domain`, `test_engine.py::test_engine_scopes_retrieval_to_the_requested_domain_and_level` (end to end) |
| RAG-006 | `src/rag/empty_retrieval_handler.py` | `test_retrieval_context.py::test_empty_collection_returns_structured_result` |
| RAG-007 | `src/rag/context_ranking.py` (flag off) | `test_retrieval_context.py::test_rank_candidates_orders_by_distance` |
| RAG-008 | `src/rag/format_calibration.py` | `test_retrieval_context.py::test_format_calibration_exact_format` |
| RAG-009 | `src/rag/build_batch_content.py` | `test_prompt_assembly.py::test_batch_content_has_one_entry_per_pair_in_order` |
| RAG-010 | `src/rag/truncate.py` | `test_prompt_assembly.py::test_oversized_prompt_is_truncated_to_the_configured_limit` |
| RAG-011 | `src/rag/system_prompt.py` (`PECS_SYSTEM_PROMPT`) | `test_prompt_assembly.py::test_system_prompt_encodes_every_criterion_and_weight` |
| RAG-012 | `src/rag/build_request.py` (`cache_control`) | `test_prompt_assembly.py::test_system_block_is_marked_for_ephemeral_caching` — **see caveat below** |
| RAG-013 | `src/rag/build_request.py` | `test_prompt_assembly.py::test_request_matches_the_messages_api_shape` |
| RAG-014 | `src/rag/llm_call.py` | `test_llm_layer.py::test_batch_of_three_is_graded_in_exactly_one_call` |
| RAG-015 | `src/rag/parse_response.py` | `test_llm_layer.py::test_fenced_json_still_parses` |
| RAG-016 | `src/rag/validate_llm_output.py` | `test_llm_layer.py::test_missing_c5_is_rejected_with_a_clear_error` |
| RAG-017 | `src/rag/retry_handler.py` | `test_llm_layer.py::test_malformed_then_valid_succeeds_within_the_budget`, `::test_a_refusal_is_not_retried`, `::test_an_empty_response_is_retried`, `::test_a_configuration_error_is_not_retried` |
| RAG-018 | `src/rag/api_error_handler.py` | `test_llm_layer.py::test_api_key_error_returns_a_structured_error_not_a_traceback` |
| RAG-019 | `src/rag/token_safety.py` | `test_prompt_assembly.py::test_oversized_batch_aborts_before_any_llm_call`, `test_engine.py::test_engine_aborts_an_oversized_batch_before_calling_the_model` |
| RAG-020 | `src/rag/usage_logger.py` | `test_llm_layer.py::test_log_usage_emits_one_line_per_call` |

## Workstream G — AI Evaluation Engine

| Task ID | Deliverable in this repo | Test |
|---|---|---|
| EVAL-001 | `src/evaluation/schema.py` | `test_evaluation_scoring.py::test_schema_matches_what_build_batch_content_produces`, `::test_normalize_qa_pair_accepts_the_documented_shapes` |
| EVAL-002 | `src/evaluation/rubric_loader.py` | `test_evaluation_scoring.py::test_editing_the_rubric_weights_changes_scoring_with_no_other_edit`, `::test_rubric_is_loaded_from_one_place_and_is_consistent`, `::test_rubric_loader_rejects_weights_that_do_not_sum_to_one` |
| EVAL-003 | `src/evaluation/rubric_version.py` | `test_evaluation_scoring.py::test_rubric_tag_changes_when_the_rubric_text_changes` |
| EVAL-004 | `src/evaluation/score_range_guard.py` | `test_evaluation_scoring.py::test_out_of_range_score_is_caught_before_scoring` |
| EVAL-005 | `src/evaluation/justification.py` | `test_evaluation_scoring.py::test_over_limit_justification_is_truncated_to_the_cap` |
| EVAL-006 | `src/evaluation/calculate_scores.py` | `test_evaluation_scoring.py::test_composite_matches_a_hand_calculation`, `::test_composite_has_no_times_ten_multiplication` |
| EVAL-007 | `src/evaluation/skill_level_map.py` | `test_evaluation_scoring.py::test_boundary_values_map_to_the_correct_band` |
| EVAL-008 | `src/evaluation/ethics_gate.py` | `test_evaluation_scoring.py::test_default_build_treats_c5_as_weighted_only`, `::test_hard_gate_caps_a_failing_c5_when_enabled` — **[DATA NEEDED], see below** |
| EVAL-009 | `src/evaluation/overall_summary.py` | `test_evaluation_scoring.py::test_summary_prefers_the_evaluator_text`, `test_engine.py::test_engine_summary_is_present_and_non_empty` |
| EVAL-010 | `src/evaluation/confidence.py` | `test_evaluation_scoring.py::test_absent_confidence_is_documented_as_not_implemented` |
| EVAL-011 | `src/evaluation/error_types.py` | `test_evaluation_scoring.py::test_two_failure_types_produce_two_distinguishable_codes` |
| EVAL-012 | `docs/evaluator_determinism.md` | n/a (documentation task) |
| EVAL-013 | `src/evaluation/model_version.py`, stamped per record in `src/evaluation/engine.py` | `test_evaluation_scoring.py::test_every_record_gets_a_non_null_model_version`, `test_engine.py::test_every_stored_record_is_self_describing` |
| EVAL-014 | `docs/evaluation_architecture.md` | n/a (documentation task; requires Member 5 review) |

## Composition (not a numbered task, required by the handoff package)

| Component | File | Test |
|---|---|---|
| Engine entry point for `POST /evaluate` | `src/evaluation/engine.py` | `test_engine.py` (20 tests) |
| Test doubles reusable by Member 6 | `tests/member4/conftest.py` | — |

## Two items that are deliberately not "green"

**EVAL-008 — C5 ethics handling.** The tracker marks this `[DATA NEEDED]`:
written confirmation from Team Beta / Alliance Compliance on whether the hard-gate
veto is active in this MVP build. That confirmation has not been received.

Implemented state: **weighted-only** (C5 contributes its 15% weight, no veto),
with the hard-gate path fully implemented behind `PECS_ETHICS_HARD_GATE` and
covered by tests. Every result records
`ethics_gate.policy_confirmation = "pending_alliance_compliance_confirmation"`.
Both behaviours are therefore ready; the *decision* is still outstanding.

**RAG-012 — Prompt caching.** Implemented correctly (ephemeral `cache_control` on
the rubric block, no volatile content in the cached prefix), but the stated
acceptance criterion — a cache-read hit on a second call — **cannot be met on
Claude Haiku 4.5 with the current prompt**: that model's minimum cacheable prefix
is 4,096 tokens and the condensed rubric is well below it, so the API silently
declines to cache. `meta.caching` reports this on every result. See
`docs/evaluation_architecture.md` for the two options (lengthen the cached prefix,
or change model) — both are team decisions, so neither was taken unilaterally.
