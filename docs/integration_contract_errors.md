# Integration Contract: Error Responses (`POST /evaluate`)

This document defines the error response contract returned by the **PECS Evaluation API** (`POST /evaluate`) when a request fails or encounters an upstream pipeline error.

---

## 1. Structured Error Response Body Shape (`EvaluationErrorResponse`)

All evaluation pipeline error responses return a standardized JSON body matching the `EvaluationErrorResponse` model:

```json
{
  "error": {
    "code": "string",
    "layer": "string",
    "message": "string",
    "details": {},
    "retryable": boolean
  }
}
```

### Field Definitions

| Field | Type | Values / Format | Description |
|---|---|---|---|
| `error.code` | `string` | e.g. `"RAG_TOKEN_SAFETY_ABORT"` | Unique, programmatic PECS error code string. |
| `error.layer` | `string` | `"rag"` or `"evaluation"` | Component layer where the error originated. |
| `error.message` | `string` | Operator description | Human-readable explanation. Never exposes raw stack traces. |
| `error.details` | `object` | Key-value dictionary | Machine-readable context (e.g. token caps, attempt counts). |
| `error.retryable` | `boolean` | `true` or `false` | Indicates whether the request may be retried by the caller. |

---

## 2. HTTP Status Code & Error Code Mapping Table

| HTTP Status Code | Status Name | Engine Error Code(s) | Origination & Cause |
|---|---|---|---|
| **403** | Forbidden | Header Auth / Middleware | Missing or invalid `x-api-key` header. |
| **413** | Content Too Large | `RAG_TOKEN_SAFETY_ABORT` | Input payload estimated tokens exceed safety cap. |
| **422** | Unprocessable Entity | `EVAL_INVALID_INPUT`, Pydantic | Missing required fields, blank whitespace, or empty submissions list. |
| **429** | Too Many Requests | `RAG_LLM_RATE_LIMITED`, Rate Limiter | Global API rate limit ($>30\text{ req}/60\text{s}$) or Anthropic API rate limit. |
| **500** | Internal Server Error | `RAG_RETRIEVAL_FAILED`, `EVAL_RUBRIC_UNAVAILABLE`, `EVAL_ETHICS_GATE_UNCONFIRMED`, Unmapped fallback | Server configuration issue, database failure, rubric file missing, or unmapped fallback error. |
| **502** | Bad Gateway | `RAG_LLM_AUTH_ERROR`, `RAG_LLM_CONNECTION_ERROR`, `RAG_LLM_BAD_REQUEST`, `RAG_LLM_SERVER_ERROR`, `RAG_LLM_REFUSAL`, `RAG_LLM_EMPTY_RESPONSE`, `RAG_LLM_UNKNOWN_ERROR`, `RAG_RESPONSE_NOT_JSON`, `RAG_RESPONSE_SCHEMA_INVALID`, `RAG_RETRY_BUDGET_EXHAUSTED`, `EVAL_SCORE_OUT_OF_RANGE`, `EVAL_MISSING_CRITERION`, `EVAL_SUBMISSION_COUNT_MISMATCH` | Upstream Anthropic LLM transport failure, malformed LLM response, or out-of-range evaluation score. |
| **504** | Gateway Timeout | `RAG_LLM_TIMEOUT` | Upstream Anthropic LLM API call timed out. |

---

## 3. Example Error Response Payloads

### 3.1 HTTP 403 Forbidden (Authentication Failure)

```json
{
  "detail": "Invalid API key."
}
```

---

### 3.2 HTTP 413 Content Too Large (Safety Abort)

```json
{
  "error": {
    "code": "RAG_TOKEN_SAFETY_ABORT",
    "layer": "rag",
    "message": "SAFETY ABORT: estimated input tokens (5200) exceed safety cap (4000)",
    "details": {
      "estimated_input_tokens": 5200,
      "safety_cap": 4000
    },
    "retryable": false
  }
}
```

---

### 3.3 HTTP 422 Unprocessable Entity (Validation Error)

```json
{
  "error": {
    "code": "EVAL_INVALID_INPUT",
    "layer": "evaluation",
    "message": "Submission at index 1 is missing required field 'student_prompt'",
    "details": {
      "submission_index": 1,
      "missing_field": "student_prompt"
    },
    "retryable": false
  }
}
```

---

### 3.4 HTTP 429 Too Many Requests (Rate Limit)

```json
{
  "error": {
    "code": "RAG_LLM_RATE_LIMITED",
    "layer": "rag",
    "message": "Rate limit exceeded on Anthropic LLM provider API",
    "details": {
      "provider": "anthropic",
      "retry_after_seconds": 5
    },
    "retryable": true
  }
}
```

---

### 3.5 HTTP 500 Internal Server Error (Server / Database Error)

```json
{
  "error": {
    "code": "RAG_RETRIEVAL_FAILED",
    "layer": "rag",
    "message": "ChromaDB vectorstore collection retrieval failed",
    "details": {
      "collection": "pels_vector_db"
    },
    "retryable": false
  }
}
```

---

### 3.6 HTTP 502 Bad Gateway (Upstream LLM / Evaluator Error)

```json
{
  "error": {
    "code": "RAG_LLM_AUTH_ERROR",
    "layer": "rag",
    "message": "Invalid Anthropic API key credential provided to evaluator",
    "details": {
      "provider": "anthropic"
    },
    "retryable": false
  }
}
```

---

### 3.7 HTTP 504 Gateway Timeout (Upstream Timeout)

```json
{
  "error": {
    "code": "RAG_LLM_TIMEOUT",
    "layer": "rag",
    "message": "Anthropic LLM call timed out after 30 seconds",
    "details": {
      "timeout_seconds": 30
    },
    "retryable": true
  }
}
```
