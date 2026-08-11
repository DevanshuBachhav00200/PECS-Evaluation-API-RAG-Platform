"""
RAG-011 — Master system prompt (PECS_SYSTEM_PROMPT).

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Encodes the C1-C6 rubric into a single condensed system prompt with a strict
JSON-only output instruction. Criterion names and anchor language follow
Member 2's ANN-002 annotation guide so the evaluator and the human annotators
are scoring the same construct; weights are imported from the canonical
ANN-001 definition rather than restated, and the skill bands come from EVAL-007.

Two properties this module must preserve:

1. **Deterministic bytes.** The prompt is assembled from constants at import
   time — no timestamps, no UUIDs, no per-request interpolation. Prompt caching
   (RAG-012) is a prefix match, so a single varying byte would invalidate the
   cache on every call.
2. **Single source.** Nothing else in the codebase may restate the rubric.
   EVAL-002 loads it from here; the API layer documents it from here.
"""

from __future__ import annotations

import hashlib

from src.dataset.annotation.annotation_schema import CRITERIA, WEIGHTS
from src.evaluation.skill_level_map import describe_bands

#: Human-readable criterion names, aligned with ANN-002.
CRITERION_NAMES = {
    "c1": "Foundations",
    "c2": "Design & Patterns",
    "c3": "Output Specification",
    "c4": "Domain Application",
    "c5": "Ethics & Safety",
    "c6": "Metacognition",
}

#: One-line scoring focus per criterion, condensed from the ANN-002 guide.
CRITERION_FOCUS = {
    "c1": (
        "Understanding of the underlying task and concepts needed to answer it. "
        "1-3 little or no usable task understanding; 4-6 partial understanding with "
        "important elements missing; 7-10 clear, appropriate understanding."
    ),
    "c2": (
        "Structure, decomposition, constraints and strategy that make the requested "
        "output actionable. 1-3 essentially no useful structure; 4-6 some structure "
        "with material gaps; 7-10 structure that materially improves execution."
    ),
    "c3": (
        "Clarity of the desired output: format, scope, depth, audience, examples, "
        "acceptance criteria. 1-3 vague or underspecified; 4-6 some useful output "
        "constraints; 7-10 specific and testable output requirements."
    ),
    "c4": (
        "Whether the prompt applies the task appropriately to its stated domain and "
        "context. 1-3 little domain adaptation; 4-6 some relevant context with "
        "notable omissions; 7-10 context that is specific, relevant and actionable."
    ),
    "c5": (
        "Handling of safety and ethical constraints appropriate to the task. Do not "
        "invent risks where none are present. 1-3 a material safety or ethical issue "
        "is mishandled where relevant; 4-6 acceptable but incomplete handling; "
        "7-10 appropriate explicit handling, or no material safety issue introduced."
    ),
    "c6": (
        "Whether the prompt asks for reflection, reasoning checks, assumptions or "
        "verification. 1-3 no meaningful reflection component; 4-6 limited checking; "
        "7-10 meaningful, actionable reflection or verification."
    ),
}


def _weight_table() -> str:
    return "\n".join(
        f"- {criterion.upper()} {CRITERION_NAMES[criterion]} "
        f"(weight {WEIGHTS[criterion] * 100:.0f}%): {CRITERION_FOCUS[criterion]}"
        for criterion in CRITERIA
    )


#: The exact JSON contract the evaluator must return. Validated by RAG-016.
OUTPUT_CONTRACT = """{
  "evaluations": [
    {
      "submission_index": 1,
      "scores": {"c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0, "c6": 0},
      "justifications": {
        "c1": "", "c2": "", "c3": "", "c4": "", "c5": "", "c6": ""
      }
    }
  ],
  "overall_summary": {
    "strength": "",
    "weakness": "",
    "improvement": ""
  }
}"""


PECS_SYSTEM_PROMPT = f"""You are the PECS evaluator. PECS is the Prompt Engineering \
Competency Standard. You grade a candidate's PROMPT — not the answer a model would \
give to that prompt — against the six PECS criteria below.

Each criterion is scored on an integer scale of 1 to 10, where 1 is the weakest and \
10 is the strongest observable performance.

CRITERIA
{_weight_table()}

SCORING RULES
- Score only what the prompt text demonstrates. Do not reward intent you cannot see \
in the text, and do not penalise a prompt for omitting something its task does not \
require.
- Judge each criterion independently. A strong C3 does not raise C6.
- The composite score is computed downstream from your six scores using the weights \
above. Do not compute, estimate, or mention a composite score, percentage, or skill \
level yourself. For reference only, the downstream bands are: {describe_bands()}.
- A calibration example may be supplied with a submission. It is a previously \
reviewed prompt with its approved C1-C6 scores and justification. Use it to align the \
severity of your scoring. It is a reference point, not a target: do not copy its \
scores, and do not assume the candidate prompt deserves the same band.
- When a submission carries no calibration example, grade it against the criteria \
above as written.

JUSTIFICATIONS
- Write one justification per criterion, at most 30 words each.
- State the specific feature of the prompt that drove the score. Avoid restating the \
criterion name and avoid generic praise or criticism.

OUTPUT FORMAT
- Return raw JSON only. No markdown, no code fences, no commentary before or after.
- Return exactly one "evaluations" entry per submission, in the order the submissions \
were given, with "submission_index" matching the index shown for that submission.
- All six criteria must be present in both "scores" and "justifications" for every \
submission. Never omit a criterion. If evidence is thin, score conservatively and say \
so in the justification.
- "scores" values are integers from 1 to 10 inclusive.
- "overall_summary" describes the batch as a whole: one shared strength, one shared \
weakness, and one concrete improvement. At most 50 words per field.

Return JSON in exactly this shape:
{OUTPUT_CONTRACT}"""


def get_system_prompt() -> str:
    """Return the master system prompt.

    A function as well as a constant so callers can be mocked in tests without
    patching module attributes.
    """
    return PECS_SYSTEM_PROMPT


def system_prompt_fingerprint() -> str:
    """Return a short stable hash of the prompt text.

    Used by EVAL-003 to detect that the rubric text changed even when the
    version string was not bumped, and by the usage logger so a logged call can
    be tied back to the exact rubric bytes that produced it.
    """
    digest = hashlib.sha256(PECS_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    return digest[:12]
