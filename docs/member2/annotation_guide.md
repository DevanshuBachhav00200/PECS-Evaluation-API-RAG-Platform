# C1–C6 Annotation Guide — Operational Draft

**Status: DRAFT — SME review required before this becomes the authoritative rubric.**

The project tracker names six criteria but the supplied artifacts do not contain the complete official low/high anchor language. Therefore this file provides an operational structure without claiming unsupported official definitions.

## Required annotation fields

For every prompt, record C1–C6 on a 1–10 scale, a concise justification, annotator ID, timestamp, confidence, and review status.

## C1 — Foundations

Assess whether the prompt demonstrates an appropriate understanding of the underlying task/concepts needed to answer it.

- Low anchor (1–3): little or no usable task understanding is expressed.
- Mid band (4–6): partial understanding is visible but important elements are missing.
- High anchor (7–10): clear, appropriate understanding is expressed.

## C2 — Design & Patterns

Assess structure, decomposition, constraints, strategy, or design choices that make the requested output actionable.

- Low anchor (1–3): essentially no useful structure or design guidance.
- Mid band (4–6): some structure or constraints, but material gaps remain.
- High anchor (7–10): strong structure/design guidance that materially improves execution.

## C3 — Output Specification

Assess how clearly the desired output is specified: format, scope, depth, audience, constraints, examples, or acceptance criteria.

- Low anchor (1–3): output is vague or underspecified.
- Mid band (4–6): some useful output constraints are present.
- High anchor (7–10): output requirements are specific and testable.

## C4 — Domain Application

Assess whether the prompt applies the task appropriately to its stated domain/context.

- Low anchor (1–3): little domain/context adaptation.
- Mid band (4–6): some relevant context but notable omissions.
- High anchor (7–10): context is specific, relevant, and actionable.

## C5 — Ethics & Safety

Assess whether the prompt handles safety/ethical constraints appropriately for the task. Do not invent risks where none are present; score the observable handling of safety/ethics requirements.

- Low anchor (1–3): material safety/ethical issue is mishandled where relevant.
- Mid band (4–6): acceptable but incomplete handling where relevant.
- High anchor (7–10): appropriate and explicit handling where relevant; otherwise no material safety issue is introduced.

## C6 — Metacognition

Assess whether the prompt asks for reflection, reasoning checks, assumptions, verification, self-evaluation, or other mechanisms that improve awareness of the solution process.

- Low anchor (1–3): no meaningful reflection/verification component.
- Mid band (4–6): limited reflection or checking.
- High anchor (7–10): meaningful, actionable reflection/verification.

## Scoring rule

Use the supplied project weights:

| Criterion | Weight |
|---|---:|
| C1 | 15% |
| C2 | 20% |
| C3 | 20% |
| C4 | 20% |
| C5 | 15% |
| C6 | 10% |

Composite score = weighted sum of C1–C6.

Skill bands:

- 0–3.99 → Weak
- 4.0–6.99 → Developing
- 7.0–10.0 → Strong

## Reviewer rule

An annotation cannot enter the approved gold set without a reviewer ID. If the reviewer disagrees, record the disagreement and reason; do not silently overwrite the original annotation.
