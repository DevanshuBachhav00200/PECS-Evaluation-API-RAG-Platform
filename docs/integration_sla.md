# Integration SLA & Performance Expectations: Timeout & Batch-Size Policy

This document defines the Service Level Agreement (SLA) status, timeout expectations, token limits, and batch-size policies for callers of the **PECS Evaluation API** (`POST /evaluate`).

---

## 1. Response-Time Bounds & Latency SLA Status

### 1.1 Benchmark Status: Pending BENCH-009

> [!IMPORTANT]
> **Performance SLA Status**: Measured response-time latency bounds and formal SLA percentiles (p50, p95, p99) are **pending official benchmark execution under task BENCH-009**.
>
> No synthetic or assumed latency numbers are published prior to completion of `BENCH-009` performance testing under representative network and load conditions.

### 1.2 Upstream LLM Timeout Behavior

- **Upstream Call Timeout**: Upstream LLM calls to Anthropic (Claude Haiku 4.5) are governed by the evaluator transport settings.
- **Timeout Response**: If an evaluation request exceeds the transport timeout, the API returns **HTTP 504 Gateway Timeout** with structured error code `RAG_LLM_TIMEOUT`.
- **Retry Recommendation**: `RAG_LLM_TIMEOUT` is flagged with `retryable: true`. Callers should implement exponential backoff before retrying timed-out requests.

---

## 2. Batch-Size Policy (API-006 Alignment)

Downstream integration callers must follow the MVP batch-size guidelines established in [`docs/batch_size_policy.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/batch_size_policy.md):

### 2.1 Enforced Request Constraints

| Constraint | Value | Enforced By | Violation Result |
|---|---|---|---|
| **Minimum Batch Size** | `1` submission | Pydantic Schema (`min_length=1`) | **HTTP 422 Unprocessable Entity** |
| **Automatic Batch Splitting** | Out of MVP Scope | API Design (API-006) | N/A (Batch processed as 1 request) |
| **Token Safety Cap** | `4000` tokens | Member 4 Token Guard (`PECS_TOKEN_SAFETY_CAP`) | **HTTP 413 Content Too Large** (`RAG_TOKEN_SAFETY_ABORT`) |

### 2.2 Practical Batch-Size Recommendations for Callers

1. **No Automatic Chunking**: The API does not automatically split large submission batches into smaller sub-batches. Every request sent to `POST /evaluate` is processed as a single atomic batch call.
2. **Recommended Practical Batch Size**: Upstream callers (Question Engine and Assessment Engine) are strongly recommended to send **small to moderate batches (1 to 5 submissions per batch)**.
3. **Preventing Token Aborts**: Submitting excessively large batches with lengthy student prompt texts risks exceeding the 4000-token safety cap (`PECS_TOKEN_SAFETY_CAP`), triggering an immediate **HTTP 413 Content Too Large** (`RAG_TOKEN_SAFETY_ABORT`).

---

## 3. Global Rate Limiting Bounds

| Limiter Property | Enforced Bound | Behavior on Exceeding |
|---|---|---|
| **Global Request Quota** | **30 requests per 60 seconds** | Returns **HTTP 429 Too Many Requests** |
| **Tracking Window** | Rolling 60-second sliding window | Timestamps older than 60s expire automatically |
| **Scope** | Global across all combined clients | Applied to `POST /evaluate` |

---

## 4. Cross-References

- **MVP Batch Size Policy**: [`docs/batch_size_policy.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/batch_size_policy.md)
- **Evaluation Architecture**: [`docs/evaluation_architecture.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/evaluation_architecture.md)
- **Request Contract**: [`docs/integration_contract_request.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/integration_contract_request.md)
- **Response Contract**: [`docs/integration_contract_response.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/integration_contract_response.md)
- **Error Contract**: [`docs/integration_contract_errors.md`](file:///c:/Users/nidhi%20chougule/Desktop/PECS-Evaluation-API-RAG-Platform-main/PECS-Evaluation-API-RAG-Platform-main/docs/integration_contract_errors.md)
