# Deployment Guide

Task: **DEP-010** · Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

The full build → run → configure → verify → deploy → rollback procedure for the
PECS Evaluation API. Written to be followable by someone who has not seen this
repository before; if a step here does not work as written, that is a bug in this
document.

Companion documents:

- `docs/deployment_config.md` — production configuration (DEP-007), **pending
  DevOps confirmation of the hosting target**
- `docs/rollback_procedure.md` — reverting a bad deploy (DEP-009)
- `docs/known_issues_and_limitations.md` — what is verified, what is not, and why

---

## 0. What this service is

One container: a FastAPI app exposing `GET /health` and `POST /evaluate`. It
embeds the candidate prompt, retrieves a calibration example from a local
ChromaDB store, sends one batched call to the Anthropic API, and returns C1–C6
scores.

Two external dependencies at runtime:

| Dependency | Required for | Failure mode without it |
|---|---|---|
| Anthropic API (`ANTHROPIC_API_KEY`) | `/evaluate` | `RAG_LLM_AUTH_ERROR` → HTTP 502 |
| ChromaDB store (mounted volume) | calibration retrieval | grades **uncalibrated** — a quiet quality drop, not an error |

The second row is the one that catches people out. An empty vector store does
not fail: RAG-006 treats "no calibration available" as a valid state, so the
service returns scores that are simply worse. **Verify the store is populated
after every deploy** (§5).

---

## 1. Prerequisites

- Docker Engine 24+ (`docker --version`) with Compose v2 (`docker compose version`)
- An Anthropic API key
- ~4 GB free disk for the image (torch and onnxruntime dominate)
- Outbound HTTPS to `api.anthropic.com`

The embedding model is baked into the image (DEP-005), so the **runtime** host
needs no HuggingFace access. The **build** host does.

---

## 2. Configure

```bash
cp .env.example .env
```

Fill in at minimum:

| Variable | Purpose | Consequence if unset |
|---|---|---|
| `ANTHROPIC_API_KEY` | evaluator credential | every `/evaluate` returns 502 |
| `API_KEY` | the `x-api-key` clients must send (API-007) | `/evaluate` fails **closed** with 500 — never open |

`CHROMA_DB_PATH` and `PECS_API_PORT` are set by the container and should not be
overridden in `.env` for containerized runs.

`.env` is excluded by both `.gitignore` and `.dockerignore`, so the credential
reaches the container only through `--env-file` at runtime and never enters an
image layer (DEP-006).

> Do not pass secrets with `-e ANTHROPIC_API_KEY=...` on the command line: the
> value lands in shell history and in `docker inspect` output. Use `--env-file`.

---

## 3. Build

```bash
docker build -t pecs-evaluation-api:1.0 .
```

Expect 5–15 minutes on a cold cache — most of it installing torch and
pre-downloading the ~90 MB embedding model. Both are cached layers afterwards, so
a source-only change rebuilds in seconds.

---

## 4. Run

Compose (recommended — it carries the volume mounts and the health check):

```bash
docker compose up -d --build
```

Or plain Docker:

```bash
docker run -d --name pecs-evaluation-api \
  -p 9000:9000 \
  --env-file .env \
  -v "$(pwd)/Database:/app/Database" \
  -v "$(pwd)/data:/app/data:ro" \
  pecs-evaluation-api:1.0
```

The two mounts are not optional:

- `Database/` is the vector store. Without the mount it lives inside the
  container's writable layer and is destroyed on the next `docker rm` (DEP-003).
- `data/` (read-only) holds the serialised embeddings the ingestion script reads.

---

## 5. Populate the vector store

**Required on first deploy, and after any change to the calibration corpus.**

```bash
docker compose exec api python -m scripts.build_vector_store
```

Expected output ends with `"ingested": 241, "collection_count": 241` (for the
corpus as of this release). The script ingests the **train split only**;
ingesting `validation` or `test` would put the benchmark's own gold answers into
calibration retrieval — see the split-leakage note in that script.

Confirm:

```bash
docker compose exec api python -c "
from src.vectorstore.client import get_client
from src.vectorstore.init import get_or_create_collection
print(get_or_create_collection(get_client()).count())"
```

A `0` here means the service is grading uncalibrated. Fix before announcing the
deploy.

---

## 6. Verify

Automated (all of DEP-003 … DEP-006 in one command):

```bash
python -m scripts.verify_container
```

Writes `benchmark/results/container_verification.json`. Exit codes: `0` all
checks passed · `1` a check failed · `2` no Docker engine, nothing was verified.

Manual smoke test:

```bash
# 1. Health — unauthenticated, must be 200
curl -fsS http://localhost:9000/health

# 2. Auth — no key must be 403, never 200
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:9000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"submissions":[{"task":"t","student_prompt":"p"}]}'

# 3. A real evaluation
curl -fsS -X POST http://localhost:9000/evaluate \
  -H "x-api-key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"submissions":[{"task":"Ask the model to review an API endpoint.",
       "student_prompt":"You are a senior reviewer. List defects by severity with a fix for each.",
       "domain":"CODE"}]}'
```

In the response from step 3, check `evaluations[0].calibration_used`. If it is
`false`, retrieval found nothing — go back to §5.

Interactive docs: `http://localhost:9000/docs`.

---

## 7. Deploy to the target host

> **DEP-007/DEP-008 status: the hosting target is not confirmed.** The tracker
> marks DEP-007 `[DATA NEEDED]` pending the DevOps Team, and no deployment has
> been performed. `docs/deployment_config.md` holds the host-specific
> configuration to be completed once the target is named; the procedure below is
> host-agnostic and is what should be run against it.

```bash
# 1. Tag for the registry
docker tag pecs-evaluation-api:1.0 <registry>/pecs-evaluation-api:1.0
docker push <registry>/pecs-evaluation-api:1.0

# 2. On the target host
docker pull <registry>/pecs-evaluation-api:1.0
docker compose up -d

# 3. Populate the store (§5) and verify (§6) against the public URL
curl -fsS https://<public-host>/health
```

Record the verification output against the deployed URL in the deployment log —
DEP-008's acceptance criterion is a real HTTP request to the deployed endpoint
returning a correctly scored response, not a successful `docker push`.

---

## 8. Operate

**Logs**

```bash
docker compose logs -f api
```

Each request emits one line with a unique id (API-011), and each completed
evaluation emits one `pecs.rag.usage` line with token counts and an estimated
USD cost (RAG-020). Aggregate the latter for cost tracking.

**Rate limiting.** 30 requests/minute globally, in-memory (API-009). It is
per-process: two replicas allow 60/minute in total, and a restart resets the
window. If a hard global limit matters, put it at the ingress, not here.

**Scaling.** Scale with replicas, not uvicorn workers — each worker loads its
own copy of the embedding model. Budget ~1.5 GB RAM per replica; the compose
file caps at 2 GB.

**Updating the corpus.** Re-run §5 against the mounted volume. No rebuild, no
redeploy.

---

## 9. Roll back

See `docs/rollback_procedure.md`. Short version:

```bash
docker compose down
docker tag pecs-evaluation-api:<previous> pecs-evaluation-api:1.0
docker compose up -d
curl -fsS http://localhost:9000/health
```

The mounted `Database/` is unaffected by an image rollback — which is the point
of keeping it outside the image.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Container never becomes healthy | model warm-load exceeds the start period | check `docker compose logs api`; raise `start_period` |
| `/evaluate` returns 500, `/health` fine | `API_KEY` unset server-side (API-007 fails closed) | set it in `.env`, restart |
| `/evaluate` returns 502 `RAG_LLM_AUTH_ERROR` | missing/invalid `ANTHROPIC_API_KEY` | fix the key; check outbound HTTPS |
| `/evaluate` returns 413 | pre-flight token cap (RAG-019) | smaller batch, or raise `PECS_TOKEN_SAFETY_CAP` deliberately |
| Scores present but `calibration_used: false` | empty vector store | §5 |
| Store empty after restart | volume not mounted | check the `-v` flag / compose `volumes` |
| Model downloads on every start | pre-baked cache missing or `HF_HOME` overridden | rebuild; confirm `HF_HOME=/opt/hf-cache` |
| `cache_read_input_tokens` always 0 | expected — see `meta.caching` | not a defect; documented in RAG-012 |
