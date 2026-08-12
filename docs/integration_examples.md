# Integration Payload Examples: PECS Evaluation API (`POST /evaluate`)

This document provides copy-pasteable request and response payload examples for common integration scenarios with the **PECS Evaluation API** (`POST /evaluate`).

Every example is strictly compliant with the official Pydantic models (`EvaluationRequest`, `EvaluationResponse`, `EvaluationErrorResponse`) and server status mappings (`ERROR_STATUS_MAP`).

---

## 1. Successful Evaluation (HTTP 200 OK)

### Request cURL Command

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

### Response Body (HTTP 200 OK)

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

## 2. Request Validation Error (HTTP 422 Unprocessable Entity)

### Request cURL Command (Missing `student_prompt` Field)

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "submissions": [
      {
        "task": "Write a Python function to sort a list."
      }
    ]
  }'
```

### Response Body (HTTP 422 Unprocessable Entity)

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "submissions",
        0,
        "student_prompt"
      ],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## 3. Authentication Error (HTTP 403 Forbidden)

### Request cURL Command (Invalid Header Key)

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: invalid-key-credential" \
  -d '{
    "submissions": [
      {
        "task": "Task description",
        "student_prompt": "Prompt text"
      }
    ]
  }'
```

### Response Body (HTTP 403 Forbidden)

```json
{
  "detail": "Invalid API key."
}
```

---

## 4. Rate-Limit Exceeded (HTTP 429 Too Many Requests)

### Response Body (HTTP 429 Too Many Requests)

Returned when request rate exceeds the global limit of 30 requests per 60 seconds:

```json
{
  "detail": "Global rate limit exceeded (max 30 requests per 60s)."
}
```

---

## 5. Token Safety Abort Error (HTTP 413 Content Too Large)

### Response Body (HTTP 413 Content Too Large)

Returned when estimated input tokens exceed the safety threshold cap (4000 tokens):

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

## 6. Upstream / Evaluator Error (HTTP 502 Bad Gateway)

### Response Body (HTTP 502 Bad Gateway)

Returned when the upstream Anthropic LLM API fails or returns invalid response data:

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

## 7. Gateway Timeout (HTTP 504 Gateway Timeout)

### Response Body (HTTP 504 Gateway Timeout)

Returned when the upstream Anthropic LLM API call times out:

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
