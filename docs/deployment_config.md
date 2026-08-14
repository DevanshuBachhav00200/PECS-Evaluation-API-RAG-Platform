# Production Deployment Configuration

Task: **DEP-007** · Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

> **`[DATA NEEDED]` — the hosting target is not confirmed.**
>
> The tracker marks DEP-007 as requiring confirmation of the current hosting
> target and the credentials owner with the **DevOps Team**. That confirmation
> has not been received, so this document specifies the configuration the service
> needs *from* a host rather than asserting which host it runs on. Every
> host-specific value below is a blank to be filled in with DevOps, not a
> default to be adopted silently.
>
> DEP-008 (deploy and verify the deployed endpoints) is blocked on this.

---

## 1. What the service requires from a host

Independent of which platform is chosen.

| Requirement | Value | Why |
|---|---|---|
| CPU | 2 vCPU minimum | the embedding model runs on CPU; ONNX/torch are the load |
| Memory | 2 GB per replica | ~1.5 GB resident with the model warm; the compose file caps at 2 GB |
| Disk | 5 GB image + store growth | the image is large (torch, onnxruntime); the store is small |
| Persistent volume | required, mounted at `/app/Database` | DEP-003 — without it the vector store dies with the container |
| Outbound HTTPS | `api.anthropic.com` | every `/evaluate` is one API call |
| Inbound | one port, default 9000 | `PECS_API_PORT` |
| TLS termination | at the ingress | the app speaks plain HTTP; `x-api-key` in cleartext over the internet is a credential leak |
| Startup grace | ≥ 90 s before the first health check | the sentence-transformer warm-load (EMB-001) |

The persistent-volume row is the one that eliminates otherwise-attractive
platforms: any host with an ephemeral filesystem will lose the vector store on
every restart, and the service will keep answering — uncalibrated. If the chosen
target cannot provide a persistent volume, the store has to move to a hosted
vector database first (VDB-012's hardening note is not in the repo — see
`docs/known_issues_and_limitations.md`), which is a change of
scope, not a configuration flag.

---

## 2. Environment

Set from the host's secret store, never baked into the image (DEP-006).

| Variable | Source | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | secret store | **[DATA NEEDED]** credentials owner — DevOps |
| `API_KEY` | secret store | the key downstream teams send as `x-api-key`; rotation owner unassigned |
| `CHROMA_DB_PATH` | fixed | `/app/Database/pels_vector_db` — must sit inside the mount |
| `PECS_API_PORT` | fixed | `9000` |
| `HF_HOME` | fixed | `/opt/hf-cache` (pre-baked model) |
| `HF_HUB_OFFLINE` | fixed | `1` — a download attempt at runtime is a misconfiguration |
| `PECS_EVALUATOR_MODEL` | default | `claude-haiku-4-5-20251001`; pinned snapshot, not an alias |
| `PECS_TOKEN_SAFETY_CAP` | default | `4000` estimated input tokens |
| `PECS_ETHICS_HARD_GATE` | **`false`** | do not enable without written Alliance Compliance confirmation (EVAL-008) |

Rotation procedure for both secrets: **[DATA NEEDED]** — who holds them, how they
are rotated, and what the rotation window is. Until that is answered, treat both
as long-lived, which is a risk worth naming rather than a decision anyone made.

---

## 3. To be completed with the DevOps Team

Blanks, deliberately left blank. Filling these in with plausible values would
produce a document that reads as confirmed and is not.

| Item | Value | Status |
|---|---|---|
| Hosting platform | ______ | **[DATA NEEDED]** |
| Region | ______ | **[DATA NEEDED]** |
| Container registry | ______ | **[DATA NEEDED]** |
| Public hostname | ______ | **[DATA NEEDED]** |
| TLS certificate management | ______ | **[DATA NEEDED]** |
| Secret store | ______ | **[DATA NEEDED]** |
| Persistent volume class / size | ______ | **[DATA NEEDED]** |
| Backup schedule for `Database/` | ______ | **[DATA NEEDED]** |
| Log aggregation target | ______ | **[DATA NEEDED]** |
| Alerting owner and on-call | ______ | **[DATA NEEDED]** |
| Replica count | ______ | **[DATA NEEDED]** |
| Ingress rate limit | ______ | see §4 |

---

## 4. Configuration decisions that are already made

These do not need DevOps input; they follow from how the service works.

**One uvicorn worker per container.** Each worker loads its own copy of the
embedding model. Scale with replicas.

**Rate limiting belongs at the ingress.** The in-process limiter (API-009) is 30
requests/minute *per process*, held in memory: N replicas allow 30×N, and a
restart resets the window. It is a safety valve against a runaway client, not a
quota mechanism. A real global limit has to sit in front of the replicas.

**Health check must be `/health`, unauthenticated.** `/evaluate` costs an API
call; using it as a liveness probe would bill for every probe. `/health` needs no
API key by design (API-004), so no credential goes into the platform's probe
config.

**No autoscaling on request rate.** A cold replica spends seconds warm-loading
the model before it can serve, so scaling out on a traffic spike adds latency
before it removes any. Scale on sustained load, with a generous cooldown.

---

## 5. Pre-deployment checklist

Every item is a command that either passes or fails; none is a judgement call.

- [ ] `pytest -q` green on the release commit
- [ ] `python -m scripts.verify_container` exits 0 (build, health, persistence, secrets)
- [ ] `python -m benchmark.run_benchmark --only retrieval` — self-retrieval pass rate recorded
- [ ] `.env` present on the host, absent from the repo and the image
- [ ] Persistent volume mounted and writable by uid 10001 (the container's non-root user)
- [ ] Vector store populated — `collection.count()` > 0 after deploy
- [ ] `/health` returns 200 over the public URL
- [ ] `/evaluate` returns a scored response with `calibration_used: true` over the public URL
- [ ] Previous image tag retained for rollback (`docs/rollback_procedure.md`)
- [ ] Downstream teams notified of the base URL and the `x-api-key` value
