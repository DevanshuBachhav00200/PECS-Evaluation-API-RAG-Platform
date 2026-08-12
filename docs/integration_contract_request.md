# Integration Contract: Evaluation Request (`POST /evaluate`)

This document defines the exact integration contract for sending evaluation request payloads from upstream services (such as the Question Engine and Assessment Engine) to the **PECS Evaluation API** (`POST /evaluate`).

---

## 1. HTTP Endpoint & Protocol Specification

- **HTTP Method**: `POST`
- **Path**: `/evaluate`
- **Content-Type**: `application/json`
- **Required Headers**:
  - `x-api-key` *(string, required)*: Authentication key matching the server's configured secret key (`API_KEY`).
  - `Content-Type: application/json` *(required)*: Specifies JSON request body payload format.

---

## 2. Request Schema Reference

The evaluation request payload is structured as a top-level JSON object containing a `submissions` array.

### 2.1 Top-Level Model: `EvaluationRequest`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `submissions` | `Array<SubmissionRequest>` | **Yes** | `min_length=1` | Non-empty list of prompt submissions to evaluate in batch. |

---

### 2.2 Item Model: `SubmissionRequest`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `task` | `string` | **Yes** | `min_length=1`, non-whitespace | The task assignment or problem description given to the student. |
| `student_prompt` | `string` | **Yes** | `min_length=1`, non-whitespace | The candidate prompt text submitted by the student to be evaluated. |
| `domain` | `string` | No *(Optional)* | String or `null` | Optional domain category (e.g. `"CODE"`) to scope calibration example retrieval. |
| `level` | `string` | No *(Optional)* | String or `null` | Optional target skill level (e.g. `"strong"`, `"developing"`, `"weak"`). |

---

## 3. Validation Rules & Enforcement

1. **Non-Empty Submissions List**: The `submissions` array must contain at least 1 item. An empty array `[]` is rejected with **HTTP 422 Unprocessable Entity**.
2. **Non-Whitespace String Enforcement**: Both `task` and `student_prompt` must be non-empty strings and cannot consist solely of whitespace characters. Blank strings or whitespace-only inputs trigger a Pydantic `ValueError` and return **HTTP 422**.
3. **Authentication Enforcement**: Requests omitting or supplying an invalid `x-api-key` header are rejected with **HTTP 403 Forbidden**.
4. **Global Rate Limiting**: The API enforces a global sliding-window rate limit of **30 requests per 60 seconds**. Requests exceeding this threshold return **HTTP 429 Too Many Requests**.

---

## 4. Valid Example Request Payload

```json
{
  "submissions": [
    {
      "task": "Write a Python function to compute the factorial of a number.",
      "student_prompt": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
      "domain": "CODE",
      "level": "strong"
    }
  ]
}
```

---

## 5. Validation Error Response Format (HTTP 422)

When request validation fails, the API returns an **HTTP 422 Unprocessable Entity** response containing structured field-level error details:

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "submissions",
        0,
        "task"
      ],
      "msg": "Value error, Field cannot be empty or whitespace only.",
      "type": "value_error"
    }
  ]
}
```
