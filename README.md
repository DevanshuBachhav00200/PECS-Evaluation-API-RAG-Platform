# PECS Evaluation RAG API

AI/ML evaluation infrastructure for the **Prompt Engineering Competency
Standard (PECS)**.

The system provides the AI/ML intelligence layer for evaluating learner
prompts using a competency-based **C1--C6 framework**.

## Team Scope

Team 2 is responsible for:

-   Dataset Construction and Engineering
-   Dataset Annotation and Validation
-   Gold / Calibration Dataset
-   Embedding Generation
-   ChromaDB Vector Storage
-   Semantic Retrieval
-   RAG Pipeline
-   AI Evaluation / Grading
-   C1--C6 Scoring
-   Evaluation and Benchmarking
-   FastAPI Integration
-   Testing
-   Dockerization
-   Deployment Preparation

## Out of Scope

The following components are owned by other teams:

-   Question Engine implementation
-   Assessment Engine implementation
-   Candidate-facing assessment workflow
-   Complete certification workflow

Team 2 provides the AI evaluation service/API consumed by downstream
components.

## Final MVP Pipeline

``` text
Dataset
   ↓
Annotation & Validation
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Semantic Retrieval
   ↓
RAG Context
   ↓
LLM Evaluation
   ↓
C1–C6 Scoring
   ↓
FastAPI
   ↓
Testing & Benchmarking
   ↓
Dockerization
   ↓
Deployment Preparation / Integration
```

## Competency Framework

The evaluation framework uses six competency dimensions:

  Code   Competency             Weight
  ------ ---------------------- --------
  C1     Foundations            15%
  C2     Design & Patterns      20%
  C3     Output Specification   20%
  C4     Domain Application     20%
  C5     Ethics & Safety        15%
  C6     Metacognition          10%

## Technology Stack

-   Python
-   Sentence Transformers
-   `all-MiniLM-L6-v2`
-   ChromaDB
-   Claude / Anthropic API
-   FastAPI
-   Pydantic
-   Pytest
-   Docker
-   Uvicorn

## Repository Structure

``` text
benchmark/    Benchmark harness, gold baseline, and generated reports
config/       Configuration
data/         Dataset lifecycle and processed artifacts
scripts/      Data, vector-store, testing, and infrastructure scripts
src/          Application source code
tests/        Automated test suite and test report
docs/         Technical, API, deployment, and limitation documentation
notebooks/    Development/analysis notebooks
```

## API

The evaluation service exposes:

  Method   Endpoint      Purpose
  -------- ------------- --------------------------------------------
  GET      `/health`     Service health check
  POST     `/evaluate`   Evaluate one or more candidate submissions

The `/evaluate` endpoint supports:

-   API-key authentication through `x-api-key`
-   Pydantic request validation
-   C1--C6 scoring
-   weighted composite scoring
-   skill-level classification
-   RAG calibration retrieval
-   ethics-gate information
-   model/rubric metadata
-   token usage and estimated cost metadata
-   structured error responses
-   request rate limiting

Interactive API documentation is available through FastAPI/Swagger at:

``` text
http://localhost:9000/docs
```

## Quick Start --- Local Development

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt -r requirements-dev.txt
```

Create the environment file:

``` bash
copy .env.example .env
```

Configure the required credentials in `.env`, including the Anthropic
API key and API key used by the service.

Build/populate the vector store:

``` bash
python -m scripts.build_vector_store
```

Start the API:

``` bash
uvicorn src.api.main:app --port 9000
```

Then open:

``` text
http://localhost:9000/docs
```

## Docker

The project includes a multi-stage Docker build.

The container:

-   uses Python 3.12
-   installs pinned dependencies in a builder stage
-   pre-bakes the `all-MiniLM-L6-v2` embedding model
-   runs as a non-root user
-   exposes port `9000`
-   includes a `/health` Docker health check
-   uses the persistent ChromaDB path `/app/Database/pels_vector_db`

Build:

``` bash
docker build -t pecs-evaluation-api:1.0 .
```

Run with the environment file and persistent database volume:

``` bash
docker run --rm -p 9000:9000 --env-file .env \
  -v "$(pwd)/Database:/app/Database" \
  pecs-evaluation-api:1.0
```

For the complete build, run, verification, deployment and rollback
procedure, see:

`docs/deployment.md`

## Testing

The final automated test suite contains:

-   **804 tests collected**
-   **804 passed**
-   **0 failed**
-   **0 errors**
-   **0 skipped**
-   **77.7 seconds wall-clock execution**
-   **82.8% overall line coverage**

The test suite covers dataset, embeddings, retrieval, vector store, RAG,
evaluation, API integration, authentication/rate limiting, regression
cases, and deployment-related checks.

Generate/regenerate the test report with:

``` bash
python -m scripts.generate_test_report
```

Run the test suite:

``` bash
pytest -q
```

The detailed report is available at:

`tests/test_report.md`

## Benchmarking

Run the benchmark harness with:

``` bash
python -m benchmark.run_benchmark
```

The current retrieval benchmark includes:

-   39-record gold baseline
-   331 serialized records
-   241 records in the active vector collection
-   Top-K retrieval evaluation
-   domain-scoping checks
-   no-match-under-scope checks

The recorded retrieval probe achieved:

-   **10/10 top-1 self-retrieval**
-   **100% top-1 self-retrieval**
-   **10/10 found within Top-K**
-   **2/2 domain-scoping checks passed**

The benchmark harness also supports grading-quality comparison, but a
live-model grading run has not produced a validated MAE/correlation
result for this release. Therefore, no grading-accuracy figure is
claimed.

Benchmark report:

`benchmark/benchmark_report.md`

## Deployment Status

The PECS Evaluation API has a documented, containerized deployment
procedure.

The deployment documentation covers:

1.  Build
2.  Configuration
3.  Container execution
4.  Vector-store population
5.  Health verification
6.  API smoke testing
7.  Deployment to a confirmed host
8.  Monitoring
9.  Rollback
10. Troubleshooting

### Current release status

**Implementation:** Completed

**Testing:** Completed

**Benchmark harness:** Completed

**Docker configuration:** Completed

**Container runtime verification:** Pending execution on a machine with
Docker Engine

**Production hosting target:** Pending final DevOps/hosting confirmation

**Production deployment:** Pending confirmed hosting target

This distinction is intentional: the software and deployment procedure
are implemented, while the final external hosting deployment requires
the target infrastructure and credentials to be confirmed.

See:

-   `docs/deployment.md`
-   `docs/deployment_config.md`
-   `docs/rollback_procedure.md`

## Known Limitations

The current MVP has several documented limitations:

1.  **Live-model grading quality has not been measured.** No validated
    MAE, correlation, consistency, or edge-case grading figure is
    claimed for the current release.
2.  **Calibration coverage is currently limited.** The active vector
    store contains CODE and EDGE records; other PECS domains require
    additional calibrated corpus coverage.
3.  **The gold baseline is based on judge consensus**, not final SME
    reference scores.
4.  **An embedding-export/vector-store schema mismatch is currently
    handled by an ingestion adapter.** The underlying serializer
    alignment remains a technical cleanup item.
5.  **Docker runtime verification is pending** because the available
    verification environment did not have a Docker Engine.
6.  **Production deployment is pending** confirmation of the final
    hosting target.
7.  **C5 ethics behaviour is currently weighted-only** pending final
    policy confirmation.
8.  **Rate limiting is in-memory and per-process** and should be moved
    to distributed infrastructure for production scaling.
9.  **Prompt caching currently records no cache hits** for the
    configured rubric prefix.
10. **Confidence signals are not implemented in the MVP** and are
    represented as unavailable.
11. **Calibration retrieval may skew toward lower-scoring examples**,
    which should be evaluated further with live grading experiments.

Full evidence and remediation guidance are documented in:

`docs/known_issues_and_limitations.md`

## Final Project Status

  Component                        Status
  -------------------------------- -------------------------------------
  Dataset Engineering              Completed
  Annotation & Validation          Completed
  Embedding Generation             Completed
  ChromaDB Vector Store            Completed
  Semantic Retrieval               Completed
  RAG Pipeline                     Completed
  AI Evaluation / Grading          Completed
  C1--C6 Scoring                   Completed
  FastAPI API                      Completed
  Authentication & Rate Limiting   Completed
  Automated Testing                Completed
  Benchmark Harness                Completed
  Dockerization                    Completed
  Deployment Procedure             Completed
  Production Deployment            Pending hosting-target confirmation

## Documentation

Key technical documents include:

-   `docs/api_reference.md`
-   `docs/deployment.md`
-   `docs/deployment_config.md`
-   `docs/rollback_procedure.md`
-   `docs/known_issues_and_limitations.md`
-   `tests/test_report.md`
-   `benchmark/benchmark_report.md`

## Final Integration Boundary

Team 2 provides the evaluation intelligence service consumed by
downstream PECS components.

``` text
Question / Assessment Engine
          ↓
Task + Candidate Prompt
          ↓
      POST /evaluate
          ↓
Embedding + Retrieval
          ↓
        RAG
          ↓
   LLM Evaluation
          ↓
      C1–C6 Scores
          ↓
 Weighted Composite Score
          ↓
   Skill-Level Result
          ↓
Structured JSON Response
```

The Question Engine, Assessment Engine, candidate-facing workflow, and
complete certification workflow remain outside Team 2's ownership.
