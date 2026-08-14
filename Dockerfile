# syntax=docker/dockerfile:1
#
# DEP-001 — PECS Evaluation API container image.
#
# Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)
#
# Multi-stage build:
#   builder  installs the pinned dependencies into a virtualenv and pre-bakes
#            the ~90 MB sentence-transformer cache (DEP-005)
#   runtime   copies both across, adds the source, and runs as a non-root user
#
# Two things this layout buys, both of which cost real time when absent:
#
# * **No compiler or build cache in the shipped image.** The wheels for torch,
#   onnxruntime and chromadb are large; keeping pip's cache and build tooling
#   out of the runtime stage is the difference between an image you can pull
#   quickly and one you cannot.
# * **No model download on cold start (DEP-005).** The embedding model is
#   fetched once at build time into HF_HOME and copied into the runtime stage.
#   Without it, every fresh container downloads ~90 MB before serving its first
#   request, and a container with no outbound network never becomes healthy at
#   all.
#
# Build and run:
#   docker build -t pecs-evaluation-api:1.0 .
#   docker run --rm -p 9000:9000 --env-file .env \
#       -v "$(pwd)/Database:/app/Database" pecs-evaluation-api:1.0
#
# See docs/deployment.md for the full build → run → verify → rollback procedure.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build-only toolchain. Present in this stage, absent from the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer, cached independently of the source so a code change does
# not reinstall torch.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# DEP-005 — pre-bake the embedding model cache so the runtime container never
# downloads it. Must match EMB-001's MODEL_NAME (src/embeddings/embedder.py);
# a mismatch here means a silent download at first request, not an error.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# PECS_API_PORT is read by the entrypoint below; the tracker specifies 9000.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    PECS_API_PORT=9000 \
    CHROMA_DB_PATH=/app/Database/pels_vector_db

# curl is used by HEALTHCHECK below and by the DEP-004 verification script.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

WORKDIR /app

# Only what the service needs at runtime. Tests, benchmarks, notebooks and raw
# data are excluded here and in .dockerignore (DEP-006).
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY requirements.txt README.md ./

# Non-root at runtime. The volume mount point is created and owned here so a
# bind-mounted Database/ is writable without the container running as root.
RUN useradd --create-home --uid 10001 pecs \
    && mkdir -p /app/Database /app/logs \
    && chown -R pecs:pecs /app /opt/hf-cache

USER pecs

EXPOSE 9000

# Unauthenticated by design (API-004) — a health check that needed the API key
# would put the credential into the container spec.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PECS_API_PORT}/health" || exit 1

# Single worker deliberately: the evaluation call is synchronous and the
# embedding model is warm-loaded per process, so each additional worker costs
# another copy of the model in RAM. Scale with replicas, not workers.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PECS_API_PORT} --workers 1"]
