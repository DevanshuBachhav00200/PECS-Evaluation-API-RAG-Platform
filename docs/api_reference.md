# PECS Evaluation API Reference Manual

This is the official, final API reference manual for the **PECS Evaluation API** service. It documents all operational endpoints, request/response models, authentication headers, error status mappings, rate-limiting policies, token safety caps, and integration examples.

---

## 1. API Overview

- **Purpose**: Provides a standardized FastAPI web interface for evaluating student prompt engineering submissions against a 6-criteria rubric (C1–C6) using a RAG (Retrieval-Augmented Generation) evaluator engine.
- **Base URL**: `http://localhost:8000` (configurable via `API_HOST` and `API_PORT` environment variables).
- **Protocol**: HTTP / HTTPS returning JSON response bodies (`application/json`).
- **Authentication**: Header-based authentication via `x-api-key`.

---

## 2. Endpoints Overview

| Endpoint Path | HTTP Method | Auth Required | Rate Limited | Summary & Description |
|---|---|---|---|---|
| `/health` | `GET` | **No** | **No** | Health check endpoint returning service operational status and warm-loaded sentence-transformer embedder availability. |
| `/evaluate` | `POST` | **Yes** (`x-api-key`) | **Yes** (30 req / 60s) | Batch evaluation endpoint scoring candidate student prompts against the PECS rubric. |

---

### 2.1 GET `/health` — Health Check

- **Method**: `GET`
- **Path**: `/health`
- **Authentication Requirement**: None (public).
- **Request Headers**: None.
- **Success Status Code**: `200 OK`
- **Response Model**: `HealthCheckResponse`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "embedder_loaded": true
}
```

---

### 2.2 POST `/evaluate` — Evaluate Submissions

- **Method**: `POST`
- **Path**: `/evaluate`
- **Authentication Requirement**: Required (`x-api-key` request header).
- **Rate Limit**: 30 requests per 60 seconds rolling window across all clients.
- **Request Headers**:
  - `x-api-key` *(string, required)*: Key credential matching the server `API_KEY`.
  - `Content-Type: application/json`
- **Request Body**: `EvaluationRequest` JSON payload.
- **Success Status Code**: `200 OK`
- **Success Response Model**: `EvaluationResponse`
- **Relevant Error Status Codes**:
  - `403 Forbidden`: Missing or invalid `x-api-key` header.
  - `413 Content Too Large`: Estimated input tokens exceed safety cap (`RAG_TOKEN_SAFETY_ABORT`).
  - `422 Unprocessable Entity`: Request body schema validation error or blank whitespace fields.
  - `429 Too Many Requests`: Global rate limit exceeded ($>30\text{ req} / 60\text{s}$) or LLM provider rate limit.
  - `500 Internal Server Error`: Unconfigured server `API_KEY`, database retrieval failure, missing rubric, or unconfirmed ethics gate.
  - `502 Bad Gateway`: Upstream Anthropic LLM transport error, invalid response schema, or score out-of-range.
  - `504 Gateway Timeout`: Upstream Anthropic LLM API call timed out (`RAG_LLM_TIMEOUT`).

---

## 3. Authentication Mechanism

Authentication is implemented via a FastAPI dependency reading the `x-api-key` request header (`src/api/auth.py`):

1. **Header Validation**:
   - If the `x-api-key` header is missing, the API returns **HTTP 403 Forbidden** (`detail: "API key header is missing."`).
   - If the `x-api-key` header is present but does not match the server's configured key, the API returns **HTTP 403 Forbidden** (`detail: "Invalid API key."`).
2. **Server Configuration Check**:
   - If the `API_KEY` environment variable is not configured or empty on the server, the dependency catches `ConfigurationError` and fails closed with **HTTP 500 Internal Server Error** (`detail: "API key authentication is not configured on the server."`).

---

## 4. Request Contract (`EvaluationRequest`)

### 4.1 Schema Definitions

#### Top-Level Request: `EvaluationRequest`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `submissions` | `Array<SubmissionRequest>` | **Yes** | `min_length=1` | Non-empty list of submissions to evaluate in batch. |

#### Item Model: `SubmissionRequest`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `task` | `string` | **Yes** | `min_length=1`, non-whitespace | Problem task prompt given to student. |
| `student_prompt` | `string` | **Yes** | `min_length=1`, non-whitespace | Candidate student prompt text to evaluate. |
| `domain` | `string` | No *(Optional)* | String or `null` | Optional domain category scope (e.g. `"CODE"`). |
| `level` | `string` | No *(Optional)* | String or `null` | Optional skill level scope (e.g. `"strong"`). |

### 4.2 Validation Rules

- **Whitespace Enforcement**: Both `task` and `student_prompt` strip whitespace. Blank or whitespace-only inputs trigger a Pydantic `ValueError` returning **HTTP 422**.
- **Batch Policy**: The API processes the array of submissions in a single atomic evaluator call. Automatic chunking or batch splitting is out of MVP scope (API-006).

---

## 5. Response Contract (`EvaluationResponse`)

### 5.1 Top-Level Response Fields

| Field | Type | Description |
|---|---|---|
| `evaluations` | `Array<SubmissionEvaluationResponse>` | Per-submission evaluation outputs. |
| `overall_summary` | `OverallSummaryResponse` | Batch-level narrative summary and criterion averages. |
| `usage` | `UsageInfoResponse` | Token usage counters and estimated USD cost. |
| `rubric_version` | `string` | Rubric version string (e.g. `"pecs-rubric-v1.0"`). |
| `rubric_tag` | `string` | Rubric tag with fingerprint (e.g. `"pecs-rubric-v1.0+abcdef123456"`). |
| `model_version` | `string` | Evaluator model version (e.g. `"claude-haiku-4-5-20251001"`). |
| `meta` | `EvaluationMetaResponse` | Batch metadata, criteria weights, and caching diagnostics. |

### 5.2 Per-Submission Evaluation Result (`SubmissionEvaluationResponse`)

- **`scores`**: Dictionary containing floats on a $0.0$ to $10.0$ scale for criteria `c1` (Clarity), `c2` (Structure), `c3` (Context), `c4` (Formatting), `c5` (Ethics), and `c6` (Complexity).
- **`composite_score`**: Weighted composite mark float on a $0.0$ to $10.0$ scale.
- **`skill_level`**: Mapped skill band (`"weak"`, `"developing"`, `"strong"`).
- **`ethics_gate`**: `EthicsGateResponse` object containing:
  - `mode`: Ethics mode (e.g. `"weighted_only"`).
  - `triggered`: `true` if ethics veto triggered, else `false`.
  - `c5_score`: Raw C5 ethics score ($0.0$–$10.0$).
  - `threshold`: Passing threshold ($3.0$).
  - `capped_score`: Capped composite score ($3.99$) if triggered, else `null`.
  - `pre_gate_composite`: Composite score prior to ethics gate logic.
  - `policy_confirmation`: Compliance confirmation code string.
- **`justifications`**: Per-criterion narrative feedback strings for criteria `c1` through `c6`.
- **`justification_truncated`**: List of criteria keys truncated due to word cap.
- **`prompt_truncated`**: Boolean indicating if student prompt was truncated during RAG assembly.
- **`calibration_used`**: Boolean indicating if calibration example was retrieved.
- **`calibration_record_id`**: Record ID string of retrieved calibration example, or `null`.
- **`domain` / `level`**: Scope strings passed at retrieval, or `null`.
- **`confidence` / `confidence_label`**: Confidence float ($0.0$–$1.0$) and categorical label, or `null`.
- **`confidence_status`**: Status string (e.g. `"not_implemented_in_mvp"`).

---

## 6. Error Contract & Mapping Table

Errors return the structured `EvaluationErrorResponse` JSON shape:

```json
{
  "error": {
    "code": "RAG_TOKEN_SAFETY_ABORT",
    "layer": "rag",
    "message": "SAFETY ABORT: estimated input tokens exceed safety cap",
    "details": { "estimated_input_tokens": 5200, "safety_cap": 4000 },
    "retryable": false
  }
}
```

### Complete `ERROR_STATUS_MAP` Reference

| Status | Engine Error Code | Description | `retryable` |
|---|---|---|---|
| **413** | `RAG_TOKEN_SAFETY_ABORT` | Input estimated tokens exceed 4000 token safety cap | `false` |
| **422** | `EVAL_INVALID_INPUT` | Invalid submission structure or missing required fields | `false` |
| **429** | `RAG_LLM_RATE_LIMITED` | Upstream Anthropic API rate limit exceeded | `true` |
| **500** | `RAG_RETRIEVAL_FAILED` | ChromaDB vectorstore database query failed | `false` |
| **500** | `EVAL_RUBRIC_UNAVAILABLE` | Evaluator rubric YAML file missing or invalid | `false` |
| **500** | `EVAL_ETHICS_GATE_UNCONFIRMED` | Alliance compliance policy confirmation failed | `false` |
| **502** | `RAG_LLM_AUTH_ERROR` | Invalid Anthropic API key credential | `false` |
| **502** | `RAG_LLM_CONNECTION_ERROR` | Transport layer connection failure to LLM API | `true` |
| **502** | `RAG_LLM_BAD_REQUEST` | Malformed request sent to Anthropic API | `false` |
| **502** | `RAG_LLM_SERVER_ERROR` | Upstream Anthropic 5xx server error | `true` |
| **502** | `RAG_LLM_REFUSAL` | LLM safety refusal triggered | `false` |
| **502** | `RAG_LLM_EMPTY_RESPONSE` | LLM returned empty text response | `true` |
| **502** | `RAG_LLM_UNKNOWN_ERROR` | Unclassified LLM client exception | `false` |
| **502** | `RAG_RESPONSE_NOT_JSON` | LLM response could not be parsed as JSON | `true` |
| **502** | `RAG_RESPONSE_SCHEMA_INVALID` | LLM response JSON failed output schema validation | `true` |
| **502** | `RAG_RETRY_BUDGET_EXHAUSTED` | All retry attempts exhausted | `false` |
| **502** | `EVAL_SCORE_OUT_OF_RANGE` | Score returned outside valid $0.0$–$10.0$ bounds | `false` |
| **502** | `EVAL_MISSING_CRITERION` | LLM evaluation omitted required criterion | `false` |
| **502** | `EVAL_SUBMISSION_COUNT_MISMATCH` | Number of evaluations returned does not match batch size | `false` |
| **504** | `RAG_LLM_TIMEOUT` | Upstream Anthropic LLM API call timed out | `true` |

---

## 7. Rate Limiting Policy (API-009)

- **Policy**: Global in-memory sliding-window rate limiter (`src/api/rate_limit.py`).
- **Quota**: **Maximum 30 requests per 60 seconds** combined across all clients.
- **Enforcement**: The 31st request within a rolling 60-second window is rejected with **HTTP 429 Too Many Requests** (`detail: "Global rate limit exceeded (max 30 requests per 60s)."`).

---

## 8. Token Safety Policy (Member 4 Token Guard)

- **Safety Threshold Cap**: `PECS_TOKEN_SAFETY_CAP = 4000` input tokens.
- **Enforcement**: Pre-flight token estimation checks total prompt length. If estimated input tokens exceed 4000, evaluation is aborted immediately.
- **Response**: Returns **HTTP 413 Content Too Large** with code `RAG_TOKEN_SAFETY_ABORT` (`retryable: false`).

---

## 9. Upstream Timeout & Transport Failures

- **Upstream Timeout**: LLM provider API calls time out if execution exceeds transport timeout limits.
- **Status & Error**: Returns **HTTP 504 Gateway Timeout** with code `RAG_LLM_TIMEOUT`.
- **Retry Guidance**: `RAG_LLM_TIMEOUT` has `retryable: true`. Callers should implement exponential backoff before retrying.

---

## 10. Sample Requests & Responses

### 10.1 Successful Batch Evaluation Example (HTTP 200 OK)

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "submissions": [
      {
        "task": "Write a Python function to check if a string is a palindrome.",
        "student_prompt": "def is_palindrome(s):\n    cleaned = \"\".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
        "domain": "CODE",
        "level": "strong"
      }
    ]
  }'
```

```json
{
  "evaluations": [
    {
      "submission_index": 1,
      "task": "Write a Python function to check if a string is a palindrome.",
      "scores": {
        "c1": 9.5,
        "c2": 9.0,
        "c3": 8.5,
        "c4": 9.0,
        "c5": 10.0,
        "c6": 8.0
      },
      "composite_score": 9.0,
      "skill_level": "strong",
      "ethics_gate": {
        "mode": "weighted_only",
        "triggered": false,
        "c5_score": 10.0,
        "threshold": 3.0,
        "capped_score": null,
        "pre_gate_composite": 9.0,
        "policy_confirmation": "pending_alliance_compliance_confirmation"
      },
      "justifications": {
        "c1": "Prompt specifies exact task requirements clearly.",
        "c2": "Logical string cleaning and slicing comparison algorithm.",
        "c3": "Sufficient context provided for execution.",
        "c4": "Clean Python syntax following PEP-8 rules.",
        "c5": "No ethical, bias, or safety violations detected.",
        "c6": "Appropriate code complexity for skill level."
      },
      "justification_truncated": [],
      "prompt_truncated": false,
      "calibration_used": true,
      "calibration_record_id": "rec-palindrome-001",
      "domain": "CODE",
      "level": "strong",
      "confidence": null,
      "confidence_label": null,
      "confidence_status": "not_implemented_in_mvp",
      "model_version": "claude-haiku-4-5-20251001",
      "rubric_version": "pecs-rubric-v1.0",
      "rubric_tag": "pecs-rubric-v1.0+abcdef123456"
    }
  ],
  "overall_summary": {
    "strength": "Clean alphanumeric filtering and concise string slicing comparison.",
    "weakness": "Does not handle extremely large input strings efficiently in memory.",
    "improvement": "Consider two-pointer approach to check palindrome in O(1) extra space.",
    "source": "evaluator",
    "truncated": [],
    "criterion_averages": {
      "c1": 9.5,
      "c2": 9.0,
      "c3": 8.5,
      "c4": 9.0,
      "c5": 10.0,
      "c6": 8.0
    }
  },
  "usage": {
    "input_tokens": 1050,
    "output_tokens": 230,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "attempts": 1,
    "estimated_input_tokens": 800,
    "cost_usd": 0.00215
  },
  "rubric_version": "pecs-rubric-v1.0",
  "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
  "model_version": "claude-haiku-4-5-20251001",
  "meta": {
    "submissions": 1,
    "calibrated_submissions": 1,
    "weights": {
      "c1": 0.15,
      "c2": 0.20,
      "c3": 0.20,
      "c4": 0.20,
      "c5": 0.15,
      "c6": 0.10
    },
    "config": {
      "PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"
    },
    "caching": {
      "requested": true,
      "prefix_tokens": 800,
      "expected_effective": false
    }
  }
}
```

---

## 11. Swagger & OpenAPI Documentation

- **Swagger UI**: Interactive browser-based API testing documentation is available at `GET /docs`.
- **OpenAPI Specification**: Raw OpenAPI 3.1.0 JSON specification is served at `GET /openapi.json`.

---

## 12. Integration References

- **Integration SLA**: [`docs/integration_sla.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/integration_sla.md)
- **Integration Payload Examples**: [`docs/integration_examples.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/integration_examples.md)
- **Question Engine Handoff**: [`docs/handoff_question_engine.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/handoff_question_engine.md)
- **Assessment Engine Handoff**: [`docs/handoff_assessment_engine.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/handoff_assessment_engine.md)

---

## 13. Known MVP Limitations

1. **No Automatic Batch Splitting (API-006 Out of MVP Scope)**: The API processes each request as a single atomic batch. Callers should manage batch sizes manually (recommended 1 to 5 submissions per batch).
2. **Fixed In-Memory Rate Limiter**: The sliding window rate limit (30 req / 60s) is managed in-memory and resets on process restart.
3. **Confidence Signals**: Evaluator confidence scores (`confidence`, `confidence_label`) are returned as `null` with `confidence_status = "not_implemented_in_mvp"`.
