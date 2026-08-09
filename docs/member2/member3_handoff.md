# Member 2 → Member 3 Handoff

## Consume

- `data/processed/annotations_normalized.csv`
- `data/processed/validation_set.csv`
- `data/processed/gold_eval_set.csv`
- `data/processed/gold_calibration_seed.json`
- `data/reports/leakage_report.csv`
- `data/reports/consistency_report.csv`
- `data/reports/3judge_kappa.csv`

## Identifier rule

Use `record_id` as the unique record key. Do not use `prompt_id` as a database primary key because the supplied cleaned dataset contains repeated `prompt_id` values with different prompt text.

## Retrieval rule

Do not mix the held-out `gold_eval_set.csv` into the ChromaDB retrieval corpus if it is used for final benchmark evaluation. Keep split metadata intact.

## Current evidence limitations

- The cleaned Member 1 dataset contains 331 approved rows: 241 train, 51 validation, 39 test.
- The cleaned set is currently concentrated in CODE plus 10 EDGE rows.
- The 88-row 3-judge consensus artifact covers EDU and CREATE at L1 and does not cover every domain in the master.
- Machine-derived rows are not represented as human-reviewed merely because they are labelled `approved` in the source workbook.

## Handoff acceptance

Before final handoff, confirm:

- [ ] SME-approved rubric guide
- [ ] Gold calibration examples reviewed
- [ ] Validation set assembled
- [ ] Gold evaluation set assembled and held out
- [ ] Zero split leakage
- [ ] Audit trail present
- [ ] Version tag created
- [ ] Member 3 acknowledges handoff
