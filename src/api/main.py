"""FastAPI application initialization and endpoints for PECS Evaluation API."""

from fastapi import FastAPI, status

from src.api.schemas import HealthCheckResponse
from src.embeddings.embedder import get_embedder

app = FastAPI(
    title="PECS Evaluation API",
    description="Evaluation API and RAG Platform for PECS",
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the operational health status of the API and asserts warm-loaded dependencies.",
    tags=["Health"],
)
def health_check() -> HealthCheckResponse:
    """Check API health status and verify warm-loaded embedder model availability."""
    embedder_ready = False
    try:
        embedder = get_embedder()
        embedder_ready = embedder is not None
    except Exception:
        embedder_ready = False

    return HealthCheckResponse(
        status="healthy" if embedder_ready else "degraded",
        version=app.version,
        embedder_loaded=embedder_ready,
    )

