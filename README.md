# PECS Evaluation RAG API

AI/ML evaluation infrastructure for the **Prompt Engineering Competency Standard (PECS)**.

The system provides the AI/ML intelligence layer for evaluating learner prompts using a competency-based C1–C6 framework.

## Team Scope

Team 2 is responsible for:

- Dataset Construction and Engineering
- Dataset Annotation and Validation
- Gold / Calibration Dataset
- Embedding Generation
- ChromaDB Vector Storage
- Semantic Retrieval
- RAG Pipeline
- AI Evaluation / Grading
- C1–C6 Scoring
- Evaluation and Benchmarking
- FastAPI Integration
- Testing
- Dockerization
- Deployment

## Out of Scope

The following components are owned by other teams:

- Question Engine implementation
- Assessment Engine implementation
- Candidate-facing assessment workflow
- Complete certification workflow

Team 2 provides the AI evaluation service/API consumed by downstream components.

## Current MVP Pipeline

Dataset
→ Annotation
→ Embeddings
→ ChromaDB
→ Semantic Retrieval
→ RAG Context
→ LLM Evaluation
→ C1–C6 Scoring
→ FastAPI
→ Testing
→ Deployment

## Competency Framework

The evaluation framework uses six competency dimensions:

| Code | Competency | Weight |
|------|------------|--------|
| C1 | Foundations | 15% |
| C2 | Design & Patterns | 20% |
| C3 | Output Specification | 20% |
| C4 | Domain Application | 20% |
| C5 | Ethics & Safety | 15% |
| C6 | Metacognition | 10% |

## Technology Stack

- Python
- Sentence Transformers
- ChromaDB
- Claude / Anthropic API
- FastAPI
- Pydantic
- Pytest
- Docker

## Repository Structure

```text
config/       Configuration
data/         Dataset lifecycle
scripts/      Data and infrastructure scripts
src/          Application source code
tests/        Automated tests
docs/         Technical documentation
notebooks/    Development/analysis notebooks
## Member 2 Implementation — Annotation, Validation & Gold Dataset

The repository now includes the Member 2 workstream (ANN-001 through ANN-012) under `src/dataset/annotation/`, with source data preserved under `data/source/member1/` and generated artifacts under `data/processed/`, `data/reports/`, and `data/audit/`.

Run the Member 2 pipeline:

```bash
python scripts/run_member2_pipeline.py
pytest -q
```

See:

- `docs/member2/TASK_MAP.md` — task-to-code mapping
- `docs/member2/annotation_guide.md` — C1–C6 annotation guidance
- `docs/member2/member3_handoff.md` — downstream handoff contract
- `docs/member2/IMPLEMENTATION.md` — execution instructions

### Current data status

The supplied cleaned dataset contains 331 rows: 241 train, 51 validation, and 39 test. The source `prompt_id` is reused in 140 rows, so Member 2 creates/preserves a stable `record_id` for downstream use. Prompt-text leakage checks are performed independently of identifier reuse.

The supplied 3-judge consensus data covers EDU and CREATE. CAREER, CODE, EDGE and GEN do not have equivalent 3-judge coverage in the supplied consensus file and therefore require SME sign-off before those examples are treated as approved gold.
