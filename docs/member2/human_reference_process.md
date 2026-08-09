# Human Reference Score Capture Process

1. Assign each reviewed record a stable `prompt_id`.
2. Store the annotator ID and reviewer ID separately.
3. Store C1–C6 scores and the written justification.
4. Record UTC timestamp and confidence.
5. Mark the source as `human_reference` only when an actual human/SME has supplied or approved the score.
6. Keep AI-generated scores (`auto_mean`, `claude_only_pending_consensus`, etc.) distinct from human reference scores.
7. If a reviewer changes a score, append an audit event containing old value, new value, reviewer, timestamp, and reason.
8. Only records with reviewer sign-off may enter the approved gold evaluation/calibration set.
