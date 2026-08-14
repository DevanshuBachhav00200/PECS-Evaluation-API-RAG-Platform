# Rollback Procedure

Task: **DEP-009** · Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

How to revert to a previous working version when a deployment introduces a
regression.

## Status of this document

The procedure below is written against the container and compose configuration
in this repository and has **not yet been exercised as a live drill** — DEP-009's
acceptance criterion is a rollback drill that successfully restores the previous
version, and that requires a deployed environment with at least two published
image tags. No hosting target has been confirmed (DEP-007, `[DATA NEEDED]`), so
there is nothing to drill against.

What has been verified is the property the procedure depends on: the vector
store lives on a mounted volume, outside the image, so replacing the image does
not touch the data (DEP-003). Run the drill in §5 as the first task once the
target host exists, and record the result here.

---

## 1. When to roll back

Roll back immediately, before investigating, if any of these is true after a
deploy:

- `/health` does not return 200 within the health-check start period
- `/evaluate` returns 5xx for a request that succeeded on the previous version
- `evaluations[].calibration_used` is `false` where it was `true` before —
  retrieval is broken, and the service is silently grading worse rather than
  erroring
- composite scores shift materially against the benchmark baseline
  (`python -m benchmark.run_benchmark --mode http --base-url <url>`) with no
  intended rubric change

The third and fourth are the ones worth stating explicitly: neither raises an
error, and both degrade the product's actual output. A deploy that is "up" is not
necessarily a deploy that is working.

**Do not** roll back for: a 403 (client sent the wrong key), a 413 (batch over
the token cap), or a 429 (rate limit). Those are the service behaving as
designed.

---

## 2. What a rollback does and does not touch

| Component | Rolled back? | Why |
|---|---|---|
| Application image | **Yes** | that is the artifact being reverted |
| Vector store (`Database/`) | No | mounted volume, outside the image (DEP-003) |
| `.env` / secrets | No | supplied at runtime, not versioned with the image |
| Calibration corpus (`data/`) | No | mounted read-only |
| Rubric version | Follows the image | it is code (EVAL-002); results record `rubric_tag` |

If the regression is in the **data** rather than the code — a bad re-ingest —
rolling back the image will not fix it. Re-ingest instead (§4).

---

## 3. Roll back the image

Prerequisite: the previous image tag still exists locally or in the registry.
Never delete the previous tag until the new one has been verified.

```bash
# 1. Confirm what is available
docker images pecs-evaluation-api

# 2. Stop the current version
docker compose down

# 3. Point the deploy tag at the previous image
docker tag pecs-evaluation-api:<previous> pecs-evaluation-api:1.0

# 4. Start
docker compose up -d

# 5. Verify — all three, in this order
curl -fsS http://localhost:9000/health
docker compose exec api python -c "
from src.vectorstore.client import get_client
from src.vectorstore.init import get_or_create_collection
print(get_or_create_collection(get_client()).count())"
curl -fsS -X POST http://localhost:9000/evaluate \
  -H "x-api-key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"submissions":[{"task":"Ask for an API review.","student_prompt":"You are a senior reviewer. List defects by severity.","domain":"CODE"}]}'
```

Step 5 is not optional and its middle command is the one people skip: a healthy
container with an empty store passes the health check and grades badly.

From a registry:

```bash
docker pull <registry>/pecs-evaluation-api:<previous>
docker tag <registry>/pecs-evaluation-api:<previous> pecs-evaluation-api:1.0
docker compose up -d
```

---

## 4. Roll back the vector store

If the corpus was re-ingested and retrieval quality regressed, restore the data,
not the image:

```bash
# Stop the service so nothing writes during the restore
docker compose stop api

# Restore from backup. No backup schedule is defined yet — VDB-012's
# persistence-hardening note is absent from the repo (see
# docs/known_issues_and_limitations.md). Until it exists, treat the
# rebuild path below as the primary recovery route.
rm -rf Database/pels_vector_db
cp -r backups/pels_vector_db-<date> Database/pels_vector_db

docker compose start api
python -m benchmark.run_benchmark --only retrieval --run-id post-rollback
```

With no backup, rebuild from the serialised embeddings — this is deterministic
and reproduces the store exactly:

```bash
docker compose exec api python -m scripts.build_vector_store --reset
```

---

## 5. The drill (to be run once a host exists)

DEP-009's acceptance criterion. Run it against staging, not production.

1. Note the current image tag and the store's record count.
2. Deploy a deliberately broken build (e.g. an image built from a commit with a
   failing `/evaluate` path).
3. Confirm the failure is visible through the checks in §1.
4. Execute §3.
5. Confirm: `/health` 200, store count unchanged, `/evaluate` returns a scored
   response with `calibration_used: true`.
6. Record the elapsed time from step 3 to step 5 here — that number is the
   service's practical recovery time, and it belongs in the SLA discussion
   (INT-004) rather than being estimated.

| Drill date | Performed by | From tag | To tag | Recovery time | Outcome |
|---|---|---|---|---|---|
| _not yet run_ | — | — | — | — | pending a confirmed host (DEP-007) |

---

## 6. Git-level rollback

If the bad change reached `main`:

```bash
# Revert the merge commit — never force-push over main (Section 7 of the tracker)
git revert -m 1 <merge-commit-sha>
git push origin main
```

Every milestone tag is a safe branch point. Rebuild and redeploy from the
reverted `main` rather than leaving the deployed image ahead of the repository —
an image whose source no longer exists on `main` cannot be debugged later.
