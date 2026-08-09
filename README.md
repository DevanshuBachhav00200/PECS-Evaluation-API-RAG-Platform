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