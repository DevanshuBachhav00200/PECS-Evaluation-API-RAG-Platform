# Member 2 Implementation

Implements ANN-001 through ANN-012 for dataset annotation, validation and gold-set preparation.

## Run

```bash
python scripts/run_member2_pipeline.py
pytest -q
```

Generated artifacts live under `data/processed/`, `data/reports/`, and `data/audit/`.

Human/SME status is preserved; AI-only scores are not silently promoted to approved gold.
