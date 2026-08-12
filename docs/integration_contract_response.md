# Integration Contract: Evaluation Response (`POST /evaluate`)

This document defines the exact integration contract for response payloads returned by the **PECS Evaluation API** (`POST /evaluate`) upon successful evaluation. Downstream consumer teams (such as frontend dashboards, analytics services, and reporting modules) can rely on these field contracts.

---

## 1. Top-Level Response Structure (`EvaluationResponse`)

A successful evaluation returns **HTTP 200 OK** with a JSON object structured as follows:

| Field | Type | Description |
|---|---|---|
| `evaluations` | `Array<SubmissionEvaluationResponse>` | List of per-submission evaluation results matching input batch order. |
| `overall_summary` | `OverallSummaryResponse` | Batch-level narrative summary of strengths, weaknesses, and recommendations. |
| `usage` | `UsageInfoResponse` | Token usage counters and estimated USD cost for the evaluation batch. |
| `rubric_version` | `string` | Version identifier of the rubric used for evaluation (e.g. `"pecs-rubric-v1.0"`). |
| `rubric_tag` | `string` | Rubric version tag with fingerprint hash (e.g. `"pecs-rubric-v1.0+abcdef123456"`). |
| `model_version` | `string` | Evaluator model identifier (e.g. `"claude-haiku-4-5-20251001"`). |
| `meta` | `EvaluationMetaResponse` | Batch-level execution metadata, weights distribution, and caching diagnostics. |

---

## 2. Per-Submission Result Schema (`SubmissionEvaluationResponse`)

Each item in the `evaluations` array represents the evaluation result for one prompt submission:

| Field | Type | Range / Format | Description |
|---|---|---|---|
| `submission_index` | `integer` | $\ge 1$ | 1-based position index of the submission in the request batch. |
| `task` | `string` | Non-empty | Original task assignment prompt text. |
| `scores` | `Dictionary<string, float>` | `c1`–`c6`: $0.0$ to $10.0$ | Individual criteria scores: `c1` (Clarity), `c2` (Structure), `c3` (Context), `c4` (Formatting), `c5` (Ethics), `c6` (Complexity). |
| `composite_score` | `float` | $0.0$ to $10.0$ | Final weighted composite score. |
| `skill_level` | `string` | `"weak"`, `"developing"`, `"strong"` | Categorical skill level band based on composite score. |
| `ethics_gate` | `EthicsGateResponse` | Object | C5 ethics gate evaluation audit block. |
| `justifications` | `Dictionary<string, string>` | `c1`–`c6` strings | Per-criterion narrative feedback justifying awarded scores. |
| `justification_truncated` | `Array<string>` | List of criteria keys | Criteria whose justification text exceeded word cap and was truncated. |
| `prompt_truncated` | `boolean` | `true` / `false` | Whether student prompt text was truncated during RAG assembly. |
| `calibration_used` | `boolean` | `true` / `false` | Whether a calibration example was retrieved from vectorstore. |
| `calibration_record_id` | `string` \| `null` | Record ID string or `null` | ID of retrieved calibration record, if used. |
| `domain` | `string` \| `null` | Domain code or `null` | Scope domain filter applied during retrieval. |
| `level` | `string` \| `null` | Level code or `null` | Scope skill level filter applied during retrieval. |
| `confidence` | `float` \| `null` | $0.0$ to $1.0$ or `null` | Evaluator confidence score (if enabled). |
| `confidence_label` | `string` \| `null` | `"low"`, `"medium"`, `"high"`, or `null` | Categorical confidence label. |
| `confidence_status` | `string` | e.g. `"not_implemented_in_mvp"` | Status of confidence signal extraction. |
| `model_version` | `string` | Version identifier | Evaluator model version string. |
| `rubric_version` | `string` | Version identifier | Rubric version string. |
| `rubric_tag` | `string` | Fingerprinted tag | Full rubric tag string. |

---

### 2.1 Ethics Gate Audit Schema (`EthicsGateResponse`)

| Field | Type | Description |
|---|---|---|
| `mode` | `string` | Ethics evaluation mode (e.g. `"weighted_only"`, `"hard_gate"`). |
| `triggered` | `boolean` | `true` if hard-gate ethics veto triggered, else `false`. |
| `c5_score` | `float` | Raw C5 ethics criterion score ($0.0$–$10.0$). |
| `threshold` | `float` | Minimum passing threshold for C5 score ($3.0$). |
| `capped_score` | `float` \| `null` | Capped composite score if ethics veto triggered ($3.99$), else `null`. |
| `pre_gate_composite` | `float` | Weighted composite score prior to ethics gate application. |
| `policy_confirmation` | `string` | Alliance Compliance policy status confirmation code. |

---

## 3. Batch Summary Schema (`OverallSummaryResponse`)

| Field | Type | Description |
|---|---|---|
| `strength` | `string` | Overall strengths narrative across evaluated submissions. |
| `weakness` | `string` | Overall weaknesses narrative identified across batch. |
| `improvement` | `string` | Actionable improvement recommendations for the student/batch. |
| `source` | `string` | Summary generation source (`"evaluator"`, `"derived"`, or `"mixed"`). |
| `truncated` | `Array<string>` | List of summary fields that were truncated. |
| `criterion_averages` | `Dictionary<string, float>` | Mean score across batch submissions for `c1` through `c6`. |

---

## 4. Usage & Cost Metrics Schema (`UsageInfoResponse`)

| Field | Type | Description |
|---|---|---|
| `input_tokens` | `integer` ($\ge 0$) | Total input tokens consumed by LLM call. |
| `output_tokens` | `integer` ($\ge 0$) | Total output tokens generated by LLM call. |
| `cache_creation_input_tokens` | `integer` ($\ge 0$) | Tokens written to ephemeral prompt cache. |
| `cache_read_input_tokens` | `integer` ($\ge 0$) | Tokens read from ephemeral prompt cache. |
| `attempts` | `integer` ($\ge 1$) | Number of LLM call attempts made (including retries). |
| `estimated_input_tokens` | `integer` ($\ge 0$) | Pre-flight token count estimation. |
| `cost_usd` | `float` ($\ge 0.0$) | Total estimated USD cost for the evaluation. |

---

## 5. Batch Metadata Schema (`EvaluationMetaResponse`)

| Field | Type | Description |
|---|---|---|
| `submissions` | `integer` ($\ge 1$) | Number of submissions evaluated in this batch. |
| `calibrated_submissions` | `integer` ($\ge 0$) | Submissions evaluated with calibration examples. |
| `weights` | `Dictionary<string, float>` | Weight distribution across criteria (`c1`–`c6`). |
| `config` | `Dictionary<string, Any>` | Evaluator model configuration settings snapshot. |
| `caching` | `Dictionary<string, Any>` | Ephemeral prompt caching diagnostics. |

---

## 6. Valid Success Example Response Payload

```json
{
  "evaluations": [
    {
      "submission_index": 1,
      "task": "Write a Python function to compute the factorial of a number.",
      "scores": {
        "c1": 9.0,
        "c2": 8.0,
        "c3": 8.0,
        "c4": 9.0,
        "c5": 10.0,
        "c6": 7.0
      },
      "composite_score": 8.55,
      "skill_level": "strong",
      "ethics_gate": {
        "mode": "weighted_only",
        "triggered": false,
        "c5_score": 10.0,
        "threshold": 3.0,
        "capped_score": null,
        "pre_gate_composite": 8.55,
        "policy_confirmation": "pending_alliance_compliance_confirmation"
      },
      "justifications": {
        "c1": "Clear task assignment scope.",
        "c2": "Correct mathematical algorithm logic.",
        "c3": "Sufficient context provided in prompt.",
        "c4": "Clean PEP-8 Python code formatting.",
        "c5": "Fully ethically compliant with no safety violations.",
        "c6": "Appropriate code complexity for task level."
      },
      "justification_truncated": [],
      "prompt_truncated": false,
      "calibration_used": true,
      "calibration_record_id": "rec-factorial-001",
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
    "strength": "Clean recursive implementation logic.",
    "weakness": "Lacks stack overflow handling for large input integers.",
    "improvement": "Consider providing an iterative loop alternative for very large inputs.",
    "source": "evaluator",
    "truncated": [],
    "criterion_averages": {
      "c1": 9.0,
      "c2": 8.0,
      "c3": 8.0,
      "c4": 9.0,
      "c5": 10.0,
      "c6": 7.0
    }
  },
  "usage": {
    "input_tokens": 1100,
    "output_tokens": 250,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "attempts": 1,
    "estimated_input_tokens": 850,
    "cost_usd": 0.00235
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
      "prefix_tokens": 850,
      "expected_effective": false
    }
  }
}
```
