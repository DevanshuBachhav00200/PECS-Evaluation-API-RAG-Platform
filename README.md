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
benchmark/    Benchmark harness, gold baseline, and generated report
config/       Configuration
data/         Dataset lifecycle
scripts/      Data and infrastructure scripts
src/          Application source code
tests/        Automated tests
docs/         Technical documentation
notebooks/    Development/analysis notebooks
```

## Quick Start

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env                                   # fill in ANTHROPIC_API_KEY and API_KEY
python -m scripts.build_vector_store                   # populate the calibration store
uvicorn src.api.main:app --port 9000                   # http://localhost:9000/docs
```

Containerized:

```bash
docker compose up -d --build
docker compose exec api python -m scripts.build_vector_store
curl -fsS http://localhost:9000/health
```

Full procedure, including deployment and rollback: [docs/deployment.md](docs/deployment.md).

## Testing and Benchmarking

```bash
pytest -q                                  # 804 tests
python -m scripts.generate_test_report     # regenerates tests/test_report.md
python -m benchmark.run_benchmark          # BENCH-003…010
python -m scripts.verify_container         # container verification (needs Docker)
```

- Test report: [tests/test_report.md](tests/test_report.md)
- Benchmark report: [benchmark/benchmark_report.md](benchmark/benchmark_report.md)

## Current State

**Read [docs/known_issues_and_limitations.md](docs/known_issues_and_limitations.md)
before relying on this build.** In particular:

- **Grading accuracy has not been measured.** The benchmark harness is complete
  and self-tested, but no run has been made against a live model, so there is no
  MAE or correlation figure for this build.
- **The calibration corpus covers one domain** (CODE, plus 10 EDGE records).
  Submissions in the other four master domains are graded without a relevant
  calibration example.
- **C5 (Ethics & Safety) is weighted-only**; it vetoes nothing, pending written
  confirmation from Alliance Compliance (EVAL-008).
- **Nothing has been deployed**, and the container has not been built — no Docker
  engine and no confirmed hosting target.