"""PECS Evaluation API package."""

from src.api.auth import verify_api_key
from src.api.config import ConfigurationError, Settings, get_settings
from src.api.logging_middleware import (
    LOG_FILE_PATH,
    RequestLoggingMiddleware,
    request_counter,
)
from src.api.main import app
from src.api.rate_limit import global_rate_limiter, rate_limit_dependency
from src.api.schemas import (
    ErrorDetailResponse,
    EthicsGateResponse,
    EvaluationErrorResponse,
    EvaluationMetaResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthCheckResponse,
    OverallSummaryResponse,
    SubmissionEvaluationResponse,
    SubmissionRequest,
    UsageInfoResponse,
)
__all__ = [
    "app",
    "verify_api_key",
    "global_rate_limiter",
    "rate_limit_dependency",
    "RequestLoggingMiddleware",
    "request_counter",
    "LOG_FILE_PATH",
    "Settings",
    "ConfigurationError",
    "get_settings",
    "SubmissionRequest",
    "EvaluationRequest",
    "EthicsGateResponse",
    "SubmissionEvaluationResponse",
    "OverallSummaryResponse",
    "UsageInfoResponse",
    "EvaluationMetaResponse",
    "EvaluationResponse",
    "ErrorDetailResponse",
    "EvaluationErrorResponse",
    "HealthCheckResponse",
]
