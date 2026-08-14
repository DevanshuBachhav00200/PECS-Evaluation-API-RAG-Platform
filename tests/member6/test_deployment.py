"""
Deployment-configuration tests (DEP-001 … DEP-006).

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

A Docker engine is not available in every environment the suite runs in, and
several DEP acceptance criteria are only fully decidable with one — ``docker
build`` completing, ``docker history`` showing no secret, a health check
answering from the host. Those are executed by
``python -m scripts.verify_container``.

What this module does is verify the properties that are decidable from the
artifacts themselves, so a regression in the container configuration is caught
by the ordinary test run rather than surviving until someone next has Docker to
hand. Concretely: that the image exposes the port the tracker specifies, that
the store path and the volume mount agree, that ``.env`` cannot enter the build
context, that the pre-baked model matches the one the code loads, and that every
variable the container needs is declared in ``.env.example``.

These are configuration invariants, not a substitute for the build. The
deployment section of the handoff package states which of the two produced each
claim.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

#: The port DEP-001 specifies.
EXPECTED_PORT = "9000"

#: Store path inside the container; must match the compose volume mount.
CONTAINER_DB_PATH = "/app/Database/pels_vector_db"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    assert DOCKERFILE.exists(), "DEP-001 deliverable missing: Dockerfile"
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> str:
    assert DOCKERIGNORE.exists(), "DEP-006 deliverable missing: .dockerignore"
    return DOCKERIGNORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> str:
    assert COMPOSE.exists(), "DEP-003 deliverable missing: docker-compose.yml"
    return COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def env_example() -> Dict[str, str]:
    assert ENV_EXAMPLE.exists(), "FOUND-003 deliverable missing: .env.example"
    settings: Dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        settings[name.strip()] = value.strip()
    return settings


# ---------------------------------------------------------------------------
# DEP-001 — the image
# ---------------------------------------------------------------------------


def test_dockerfile_is_multi_stage(dockerfile: str) -> None:
    """A single-stage image ships the compiler and pip cache with the service."""
    assert len(re.findall(r"^FROM ", dockerfile, flags=re.MULTILINE)) >= 2


def test_dockerfile_exposes_the_specified_port(dockerfile: str) -> None:
    assert f"EXPOSE {EXPECTED_PORT}" in dockerfile


def test_dockerfile_installs_the_pinned_requirements(dockerfile: str) -> None:
    assert "requirements.txt" in dockerfile
    assert "pip install -r requirements.txt" in dockerfile


def test_dockerfile_runs_as_a_non_root_user(dockerfile: str) -> None:
    """A container running as root writes root-owned files into the host mount."""
    assert re.search(r"^USER (?!root)", dockerfile, flags=re.MULTILINE)


def test_dockerfile_declares_a_health_check(dockerfile: str) -> None:
    """The compose health gate and DEP-004's verification both rely on it."""
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile


def test_health_check_does_not_carry_the_api_key(dockerfile: str) -> None:
    """/health is unauthenticated (API-004) precisely so this stays true."""
    healthcheck = dockerfile[dockerfile.index("HEALTHCHECK") :]
    assert "x-api-key" not in healthcheck
    assert "API_KEY" not in healthcheck.split("CMD", 1)[1].split("\n")[0]


def test_entrypoint_serves_the_fastapi_app(dockerfile: str) -> None:
    assert "uvicorn" in dockerfile
    assert "src.api.main:app" in dockerfile


def test_image_does_not_copy_tests_or_data(dockerfile: str) -> None:
    """Runtime image carries the service, not the suite or the corpus."""
    copies = re.findall(r"^COPY\s+(?!--from)(.+)$", dockerfile, flags=re.MULTILINE)
    copied = " ".join(copies)
    assert "tests/" not in copied
    assert "data/" not in copied
    assert "benchmark/" not in copied


# ---------------------------------------------------------------------------
# DEP-005 — model cache pre-baking
# ---------------------------------------------------------------------------


def test_dockerfile_prebakes_the_embedding_model(dockerfile: str) -> None:
    assert "SentenceTransformer(" in dockerfile
    assert "HF_HOME" in dockerfile


def test_prebaked_model_matches_the_one_the_code_loads(dockerfile: str) -> None:
    """A drifted string here means a silent ~90 MB download on cold start.

    Read from the source of truth rather than duplicated, so changing
    ``MODEL_NAME`` without updating the Dockerfile fails this test.
    """
    model_name = _model_name_from_source()
    assert f"SentenceTransformer('{model_name}')" in dockerfile.replace('"', "'")


def _model_name_from_source() -> str:
    source = (REPO_ROOT / "src" / "embeddings" / "embedder.py").read_text(encoding="utf-8")
    match = re.search(r'MODEL_NAME:\s*str\s*=\s*"([^"]+)"', source)
    assert match, "could not read MODEL_NAME from src/embeddings/embedder.py"
    return match.group(1)


def test_compose_disables_hub_network_access(compose: str) -> None:
    """With the cache pre-baked, a download attempt is a bug worth failing on."""
    assert "HF_HUB_OFFLINE" in compose


# ---------------------------------------------------------------------------
# DEP-002 — environment injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variable", ["ANTHROPIC_API_KEY", "API_KEY", "CHROMA_DB_PATH"]
)
def test_required_variable_is_declared_in_env_example(
    env_example: Dict[str, str], variable: str
) -> None:
    assert variable in env_example


def test_env_example_ships_no_real_secret(env_example: Dict[str, str]) -> None:
    """Placeholders only — a filled-in example is a committed credential."""
    for name in ("ANTHROPIC_API_KEY", "API_KEY"):
        assert env_example[name] == "", f"{name} must be empty in .env.example"

    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "sk-ant" not in text


def test_compose_injects_the_environment_from_a_file(compose: str) -> None:
    assert "env_file" in compose
    assert ".env" in compose


def test_container_port_is_configurable_and_defaults_correctly(
    dockerfile: str, compose: str
) -> None:
    assert f"PECS_API_PORT={EXPECTED_PORT}" in dockerfile
    assert f'"{EXPECTED_PORT}:{EXPECTED_PORT}"' in compose


# ---------------------------------------------------------------------------
# DEP-003 — persistence
# ---------------------------------------------------------------------------


def test_compose_mounts_the_vector_store_as_a_volume(compose: str) -> None:
    assert "./Database:/app/Database" in compose


def test_store_path_and_mount_point_agree(dockerfile: str, compose: str) -> None:
    """A store path outside the mount persists nothing and reports no error."""
    assert CONTAINER_DB_PATH in dockerfile
    assert CONTAINER_DB_PATH in compose
    assert CONTAINER_DB_PATH.startswith("/app/Database/")


def test_corpus_is_mounted_read_only(compose: str) -> None:
    assert "./data:/app/data:ro" in compose


def test_compose_health_check_allows_for_the_model_warm_load(compose: str) -> None:
    """A start period shorter than the warm load marks a healthy container down."""
    match = re.search(r"start_period:\s*(\d+)s", compose)
    assert match, "compose health check declares no start_period"
    assert int(match.group(1)) >= 60


# ---------------------------------------------------------------------------
# DEP-006 — secrets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", [".env", "*.pem", "*.key", "secrets/"])
def test_dockerignore_excludes_secret_material(dockerignore: str, pattern: str) -> None:
    assert pattern in dockerignore


def test_dockerignore_keeps_the_example_env(dockerignore: str) -> None:
    """The example is documentation, and must survive the .env* exclusion."""
    assert "!.env.example" in dockerignore


@pytest.mark.parametrize("pattern", [".git", ".venv/", "tests/", "data/", "Database/"])
def test_dockerignore_excludes_build_noise(dockerignore: str, pattern: str) -> None:
    assert pattern in dockerignore


def test_no_env_file_is_committed() -> None:
    """The strongest form of DEP-006: the secret file does not exist in the repo."""
    tracked = REPO_ROOT / ".env"
    if tracked.exists():
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^\.env$", gitignore, flags=re.MULTILINE), (
            "a .env exists and .gitignore does not exclude it"
        )


def test_settings_repr_does_not_leak_secret_values() -> None:
    """Config objects end up in logs and error paths; they must self-redact."""
    from src.api.config import Settings

    settings = Settings(
        env={
            "ANTHROPIC_API_KEY": "sk-ant-should-never-print",
            "API_KEY": "server-secret-should-never-print",
            "CHROMA_DB_PATH": CONTAINER_DB_PATH,
        }
    )
    rendered = repr(settings)

    assert "sk-ant-should-never-print" not in rendered
    assert "server-secret-should-never-print" not in rendered
    assert CONTAINER_DB_PATH in rendered


def test_missing_required_variable_fails_fast_with_a_clear_message() -> None:
    """API-012: startup must fail loudly, not serve without a credential."""
    from src.api.config import ConfigurationError, Settings

    settings = Settings(env={"API_KEY": "x", "CHROMA_DB_PATH": "/tmp/db"})
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        settings.validate()


# ---------------------------------------------------------------------------
# Verification tooling
# ---------------------------------------------------------------------------


def test_container_verification_script_exists_and_declares_every_dep_check() -> None:
    """DEP-003…006 each need an executable check, not a written assurance."""
    source = (REPO_ROOT / "scripts" / "verify_container.py").read_text(encoding="utf-8")
    for task in ("DEP-001", "DEP-003", "DEP-004", "DEP-005", "DEP-006"):
        assert task in source


def test_verification_script_reports_absence_rather_than_simulating(
) -> None:
    """Without Docker it must say so, not emit a green result nobody earned."""
    source = (REPO_ROOT / "scripts" / "verify_container.py").read_text(encoding="utf-8")
    assert "docker_unavailable" in source
    assert "It does\nnot simulate the checks." in source or "not simulate" in source
