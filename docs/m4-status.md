# M4 Status

Status snapshot: 2026-10-03.

M4 — Real-world Generalization is **in progress**.

## Starting evidence

M3 established a review-all acceptance policy on the current held-out real document. The policy has 0% accepted error and 0% automation because no measured threshold safely supports automatic acceptance.

The current held-out slice has no unseen-writer decisions, and capture conditions are not yet classified in evaluation metadata. These are the two primary M4 evidence gaps.

## Work queue

- classify reviewed documents by supported capture condition;
- create an explicit unseen-writer evaluation slice without document or writer leakage;
- generate per-condition and unseen-writer metrics;
- document each observed generalization failure mode and confirm that it routes to review or unresolved;
- verify that the M3 accepted-error gate still holds.
