# Member 2 Task Map (ANN-001 → ANN-012)

| Task | Implementation | Artifact |
|---|---|---|
| ANN-001 | `src/dataset/annotation/annotation_schema.py` | canonical annotation record |
| ANN-002 | `docs/member2/annotation_guide.md` | C1-C6 annotator guide |
| ANN-003 | `src/dataset/annotation/consistency.py` | consistency flags/report |
| ANN-004 | `src/dataset/annotation/normalize.py` + `validation.py` | 1-10 validation |
| ANN-005 | `src/dataset/annotation/scoring.py` | weighted score + skill band |
| ANN-006 | `scripts/run_member2_pipeline.py` | `gold_calibration_seed.json` |
| ANN-007 | `docs/member2/human_reference_process.md` | human-reference process |
| ANN-008 | `src/dataset/annotation/review.py` | second-pass review rules |
| ANN-009 | `src/dataset/annotation/disagreement.py` | disagreement report |
| ANN-010 | `src/dataset/annotation/agreement.py` | 3-judge agreement report |
| ANN-011 | `src/dataset/annotation/validation.py` + pipeline | validation/gold evaluation sets + leakage report |
| ANN-012 | `src/dataset/annotation/audit.py` | append-only audit events |

## Member 3 handoff

See `docs/member2/member3_handoff.md`.

Use `record_id` as the stable downstream key. Preserve `prompt_id` as source metadata because it is not guaranteed unique in the supplied data.
