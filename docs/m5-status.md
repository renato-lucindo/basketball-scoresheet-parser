# M5 Status

Status snapshot: 2026-10-03.

M5 — Efficient Human Review is **complete**.

## Review workflow

The contributor-facing [Human Review Workflow](review-workflow.md) documents the complete local process:

1. prepare and import geometry review;
2. build stable crops;
3. review scoring, jersey, and foul decisions;
4. complete the deterministic second-review sample;
5. verify readiness before training;
6. regenerate held-out quality metrics after model or threshold changes.

Accepted and adjusted decisions return to `manifest.reviewed.jsonl` with final labels and reviewed bounding boxes. The jersey and foul training commands use this reusable reviewed manifest by default.

## Integrity controls

Review imports reject stale, contradictory, or invalid data before updating reusable records. Enforced checks cover:

- source-image and crop-image hashes;
- regenerated crops;
- duplicate tasks, documents, regions, and crop IDs;
- unknown or wrong-stage records;
- invalid decisions, labels, dimensions, and bounding boxes;
- records outside the audit sample.

Automated tests cover reusable adjusted labels, second-review completion, stale source rejection, duplicate task rejection, and invalid label rejection.

## Efficiency history

[`evaluation/review-efficiency.json`](../evaluation/review-efficiency.json) records comparable M3 and M4 snapshots with corpus identity, decision count, accepted error, automation, review rate, candidate coverage, and the source report.

The review rate remains 100% because no global threshold has independent evidence that satisfies the quality gate. Candidate coverage is reported separately so increased model output cannot be mistaken for safe automation.

## Exit-criterion evidence

- Review-required decisions move through a documented workflow and return as reusable reviewed data.
- Review and automation rates are recorded across milestone snapshots.
- No threshold or model result is promoted when the accepted-error gate fails or lacks independent evidence.
- Stale, contradictory, and invalid review artifacts are detected before reuse.

M6 can now focus on the reproducible v1.0 release contract and clean-checkout validation.
