"""
Container build, health, persistence and secrets verification.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)
Covers: DEP-004 (local build + health check), DEP-003 (persistence across a
restart), DEP-005 (no model re-download on a second start), DEP-006 (no secret
in the image layers).

Each check is the executable form of one task's acceptance criterion, so
"verified" means a command was run and its output recorded — not that someone
read the Dockerfile and agreed with it. Results are written to
``benchmark/results/container_verification.json`` so the deployment section of
the handoff package cites a run rather than an assertion.

Requires a working Docker engine. When Docker is absent the script reports
``docker_unavailable`` per check and exits non-zero, so a CI job or a handoff
review can tell "not verified here" apart from "verified and passing". It does
not simulate the checks.

Usage::

    python -m scripts.verify_container                    # all checks
    python -m scripts.verify_container --check-secrets    # DEP-006 only
    python -m scripts.verify_container --keep             # leave the container up
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "benchmark" / "results" / "container_verification.json"

IMAGE_TAG = "pecs-evaluation-api:verify"
CONTAINER_NAME = "pecs-verify"
HOST_PORT = 9000

#: Environment variable names whose *values* must never appear in image layers.
SECRET_ENV_NAMES = ("ANTHROPIC_API_KEY", "API_KEY")


class DockerUnavailable(RuntimeError):
    """Raised when no usable Docker engine is present."""


def _docker() -> str:
    binary = shutil.which("docker")
    if not binary:
        raise DockerUnavailable(
            "docker was not found on PATH. Install Docker Desktop (Windows/macOS) "
            "or docker-ce (Linux), start the engine, and re-run."
        )

    probe = subprocess.run(
        [binary, "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise DockerUnavailable(
            "the docker CLI is installed but the engine is not reachable: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )

    return binary


def _run(args: Sequence[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)


def _result(
    check: str, task: str, passed: Optional[bool], detail: str, **extra: Any
) -> Dict[str, Any]:
    return {
        "check": check,
        "task": task,
        "passed": passed,
        "detail": detail.strip(),
        **extra,
    }


# ---------------------------------------------------------------------------
# DEP-001 / DEP-004 — build and health
# ---------------------------------------------------------------------------


def check_build(docker: str) -> Dict[str, Any]:
    """DEP-001 — `docker build` completes with no errors from a clean checkout."""
    started = time.perf_counter()
    completed = _run([docker, "build", "-t", IMAGE_TAG, str(REPO_ROOT)])
    elapsed = round(time.perf_counter() - started, 1)

    return _result(
        "build",
        "DEP-001",
        completed.returncode == 0,
        completed.stderr[-2000:] if completed.returncode else "build succeeded",
        image=IMAGE_TAG,
        seconds=elapsed,
    )


def _wait_for_health(docker: str, attempts: int = 40, delay: float = 5.0) -> Dict[str, Any]:
    """Poll the container's own health status until it settles."""
    for attempt in range(1, attempts + 1):
        completed = _run(
            [docker, "inspect", "--format", "{{.State.Health.Status}}", CONTAINER_NAME],
            timeout=60,
        )
        status = completed.stdout.strip()

        if status == "healthy":
            return {"status": status, "attempts": attempt}
        if status == "unhealthy":
            return {"status": status, "attempts": attempt}

        time.sleep(delay)

    return {"status": "timeout", "attempts": attempts}


def check_health(docker: str, env_file: Optional[Path], keep: bool) -> List[Dict[str, Any]]:
    """DEP-004 — run the image and confirm /health answers from the outside."""
    _run([docker, "rm", "-f", CONTAINER_NAME], timeout=120)

    database_dir = REPO_ROOT / "Database"
    database_dir.mkdir(parents=True, exist_ok=True)

    args = [
        docker, "run", "-d", "--name", CONTAINER_NAME,
        "-p", f"{HOST_PORT}:9000",
        "-v", f"{database_dir}:/app/Database",
    ]
    if env_file and env_file.exists():
        args += ["--env-file", str(env_file)]
    args.append(IMAGE_TAG)

    started = _run(args, timeout=300)
    if started.returncode != 0:
        return [
            _result("container_start", "DEP-004", False, started.stderr[-2000:]),
            _result("health_endpoint", "DEP-004", False, "container did not start"),
        ]

    results = [_result("container_start", "DEP-004", True, "container started")]

    health = _wait_for_health(docker)
    results.append(
        _result(
            "container_health",
            "DEP-004",
            health["status"] == "healthy",
            f"docker health status: {health['status']} after {health['attempts']} poll(s)",
            **health,
        )
    )

    # The acceptance criterion names curl against the exposed port, so the check
    # is made from the host rather than from inside the container.
    probe = _run(
        [
            docker, "run", "--rm", "--network", "host", "curlimages/curl:latest",
            "-fsS", f"http://127.0.0.1:{HOST_PORT}/health",
        ],
        timeout=120,
    )
    body = probe.stdout.strip()
    results.append(
        _result(
            "health_endpoint",
            "DEP-004",
            probe.returncode == 0 and '"status"' in body,
            body or probe.stderr[-500:],
            body=body[:500],
        )
    )

    # DEP-005 — a second start must not re-download the embedding model.
    _run([docker, "restart", CONTAINER_NAME], timeout=300)
    _wait_for_health(docker)
    logs = _run([docker, "logs", CONTAINER_NAME], timeout=120)
    combined = (logs.stdout + logs.stderr).lower()
    downloading = any(
        marker in combined for marker in ("downloading", "fetching 1 files", "%|")
    )
    results.append(
        _result(
            "model_cache_prebaked",
            "DEP-005",
            not downloading,
            "no download markers in container logs on second start"
            if not downloading
            else "container logs show a model download on restart",
        )
    )

    if not keep:
        _run([docker, "rm", "-f", CONTAINER_NAME], timeout=120)

    return results


# ---------------------------------------------------------------------------
# DEP-003 — persistence across a restart
# ---------------------------------------------------------------------------


def check_persistence(docker: str) -> Dict[str, Any]:
    """DEP-003 — data written before a restart is still present afterwards."""
    database_dir = REPO_ROOT / "Database"
    database_dir.mkdir(parents=True, exist_ok=True)

    probe = (
        "from src.vectorstore.client import get_client; "
        "from src.vectorstore.init import get_or_create_collection; "
        "c = get_or_create_collection(get_client()); "
        "c.upsert(ids=['dep003-probe'], embeddings=[[0.5]*384], "
        "documents=['persistence probe'], "
        "metadatas=[{'domain':'CODE','level':'strong','c1':8,'c2':7,'c3':9,"
        "'c4':6,'c5':10,'c6':5,'source':'train','version':'pecs-v1.0'}]); "
        "print(c.count())"
    )
    verify = (
        "from src.vectorstore.client import get_client; "
        "from src.vectorstore.init import get_or_create_collection; "
        "c = get_or_create_collection(get_client()); "
        "print(c.get(ids=['dep003-probe'])['ids'])"
    )

    mount = ["-v", f"{database_dir}:/app/Database"]

    write = _run(
        [docker, "run", "--rm", *mount, IMAGE_TAG, "python", "-c", probe], timeout=600
    )
    if write.returncode != 0:
        return _result("persistence", "DEP-003", False, write.stderr[-1500:])

    # A separate container, same mount — the strongest form of "survives a
    # restart", since nothing of the first container remains.
    read = _run(
        [docker, "run", "--rm", *mount, IMAGE_TAG, "python", "-c", verify], timeout=600
    )

    return _result(
        "persistence",
        "DEP-003",
        "dep003-probe" in read.stdout,
        read.stdout.strip() or read.stderr[-1500:],
    )


# ---------------------------------------------------------------------------
# DEP-006 — no secret baked into the image
# ---------------------------------------------------------------------------


def check_secrets(docker: str, env_file: Optional[Path]) -> List[Dict[str, Any]]:
    """DEP-006 — `docker history` shows no secret values, and .env is absent."""
    results: List[Dict[str, Any]] = []

    history = _run(
        [docker, "history", "--no-trunc", "--format", "{{.CreatedBy}}", IMAGE_TAG],
        timeout=300,
    )

    secret_values: List[str] = []
    if env_file and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            name, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if name.strip() in SECRET_ENV_NAMES and len(value) >= 8:
                secret_values.append(value)

    leaked = [value for value in secret_values if value in history.stdout]
    results.append(
        _result(
            "history_has_no_secret_values",
            "DEP-006",
            not leaked,
            "no secret value found in image history"
            if not leaked
            else f"{len(leaked)} secret value(s) present in image layers",
            secrets_checked=len(secret_values),
        )
    )

    # A .env inside the image would be readable regardless of what history shows.
    listing = _run(
        [docker, "run", "--rm", "--entrypoint", "sh", IMAGE_TAG, "-c", "ls -a /app"],
        timeout=300,
    )
    results.append(
        _result(
            "env_file_absent_from_image",
            "DEP-006",
            ".env" not in listing.stdout.split(),
            listing.stdout.strip()[:500],
        )
    )

    # Belt and braces: the build context itself must not carry the file.
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    results.append(
        _result(
            "dockerignore_excludes_env",
            "DEP-006",
            "\n.env\n" in f"\n{dockerignore}\n",
            ".dockerignore excludes .env",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Container verification (DEP-003…006)")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--check-health", action="store_true")
    parser.add_argument("--check-persistence", action="store_true")
    parser.add_argument("--check-secrets", action="store_true")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--keep", action="store_true", help="leave the container running")
    args = parser.parse_args(argv)

    selected = {
        "build": args.check_build,
        "health": args.check_health,
        "persistence": args.check_persistence,
        "secrets": args.check_secrets,
    }
    if not any(selected.values()):
        selected = {key: True for key in selected}

    try:
        docker = _docker()
    except DockerUnavailable as exc:
        report = {
            "status": "docker_unavailable",
            "reason": str(exc),
            "checks": [
                _result(name, "DEP-00x", None, "not run — docker unavailable")
                for name, wanted in selected.items()
                if wanted
            ],
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    checks: List[Dict[str, Any]] = []

    if selected["build"] or selected["health"] or selected["persistence"] or selected["secrets"]:
        # Every other check needs the image, so build unconditionally.
        checks.append(check_build(docker))
        if not checks[-1]["passed"]:
            report = {"status": "build_failed", "checks": checks}
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 1

    if selected["health"]:
        checks.extend(check_health(docker, args.env_file, args.keep))
    if selected["persistence"]:
        checks.append(check_persistence(docker))
    if selected["secrets"]:
        checks.extend(check_secrets(docker, args.env_file))

    failed = [check for check in checks if check["passed"] is False]
    report = {
        "status": "passed" if not failed else "failed",
        "image": IMAGE_TAG,
        "checks": checks,
        "failed_checks": [check["check"] for check in failed],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main(sys.argv[1:]))
