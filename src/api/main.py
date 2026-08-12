"""FastAPI application initialization and endpoints for PECS Evaluation API."""

from typing import Dict, Union

from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse

from src.api.auth import verify_api_key
from src.api.rate_limit import rate_limit_dependency
from src.api.schemas import (
    EvaluationErrorResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthCheckResponse,
)
from src.embeddings.embedder import get_embedder
from src.evaluation.engine import evaluate_batch_safe

app = FastAPI(
    title="PECS Evaluation API",
    description="Evaluation API and RAG Platform for PECS",
    version="1.0.0",
)

#: Error code to HTTP status code mapping (API-010).
ERROR_STATUS_MAP: Dict[str, int] = {
    "EVAL_INVALID_INPUT": 422,
    "RAG_TOKEN_SAFETY_ABORT": 413,
    "RAG_LLM_AUTH_ERROR": status.HTTP_502_BAD_GATEWAY,
    "RAG_LLM_RATE_LIMITED": status.HTTP_429_TOO_MANY_REQUESTS,
    "RAG_LLM_TIMEOUT": status.HTTP_504_GATEWAY_TIMEOUT,
    "RAG_LLM_CONNECTION_ERROR": status.HTTP_502_BAD_GATEWAY,
    "RAG_LLM_SERVER_ERROR": status.HTTP_502_BAD_GATEWAY,
    "RAG_LLM_REFUSAL": status.HTTP_502_BAD_GATEWAY,
    "RAG_LLM_EMPTY_RESPONSE": status.HTTP_502_BAD_GATEWAY,
    "RAG_LLM_UNKNOWN_ERROR": status.HTTP_502_BAD_GATEWAY,
    "RAG_RESPONSE_NOT_JSON": status.HTTP_502_BAD_GATEWAY,
    "RAG_RESPONSE_SCHEMA_INVALID": status.HTTP_502_BAD_GATEWAY,
    "RAG_RETRY_BUDGET_EXHAUSTED": status.HTTP_502_BAD_GATEWAY,
    "RAG_RETRIEVAL_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "EVAL_SCORE_OUT_OF_RANGE": status.HTTP_502_BAD_GATEWAY,
    "EVAL_MISSING_CRITERION": status.HTTP_502_BAD_GATEWAY,
    "EVAL_SUBMISSION_COUNT_MISMATCH": status.HTTP_502_BAD_GATEWAY,
    "EVAL_RUBRIC_UNAVAILABLE": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "EVAL_ETHICS_GATE_UNCONFIRMED": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


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


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Prompt Submissions",
    description="Evaluates a batch of candidate prompt submissions using the PECS RAG evaluation engine.",
    tags=["Evaluation"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit_dependency)],
    responses={
        200: {"model": EvaluationResponse, "description": "Successful batch evaluation result."},
        403: {"description": "Missing or invalid API key."},
        413: {"model": EvaluationErrorResponse, "description": "Token safety limit exceeded."},
        422: {"model": EvaluationErrorResponse, "description": "Invalid submission payload."},
        429: {"description": "Global rate limit exceeded (max 30 requests per 60s)."},
        500: {"model": EvaluationErrorResponse, "description": "Internal evaluator or server configuration error."},
        502: {"model": EvaluationErrorResponse, "description": "Bad gateway or upstream LLM failure."},
        504: {"model": EvaluationErrorResponse, "description": "Upstream LLM timeout."},
    },
)
def evaluate(request: EvaluationRequest) -> Union[EvaluationResponse, JSONResponse]:
    """Evaluate a batch of prompt submissions and return rubric scores and justifications."""
    qa_pairs = request.to_qa_pairs()
    result = evaluate_batch_safe(qa_pairs)

    if "error" in result:
        code = result["error"].get("code", "")
        status_code = ERROR_STATUS_MAP.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        return JSONResponse(status_code=status_code, content=result)

    return EvaluationResponse.model_validate(result)


