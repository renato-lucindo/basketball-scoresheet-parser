# Project Goals

## Mission

Basketball Scoresheet Parser turns supported basketball scoresheets into structured, reliable, and auditable data. The project starts with the FECABA scoresheet format and automates only the decisions that meet an explicit confidence and quality threshold. Ambiguous information must remain visible and be routed to review instead of being silently guessed or corrected.

## Product Principle

Correctness comes before automation rate.

Every extracted value that depends on uncertain visual or handwriting recognition should preserve enough evidence to explain the decision, expose confidence when applicable, and use an explicit status such as accepted, review, or unresolved.

## Goals Through v1.0

1. Make the FECABA pipeline work end to end from a supported image or PDF to a structured `DocumentResult`/JSON result.
2. Extract the core game record: teams, players, participation, starters, scoring by period, scoring events, individual fouls, team fouls, and final scores.
3. Keep uncertain data reviewable. Low-confidence or contradictory observations must never become silently accepted facts.
4. Increase automation only when quality gates remain satisfied on held-out real data.
5. Evaluate generalization on documents and, when writer identity is available, handwriting that was not used for training.
6. Keep the dataset, review, training, evaluation, and parsing workflow reproducible for external contributors.
7. Maintain a stable, documented output model so downstream consumers can rely on the parser without depending on internal implementation details.
8. Keep the project healthy as open source through automated tests, CI, contributor documentation, and versioned releases.

## v1.0 Quality Gates

The FECABA v1.0 milestone is complete only when all of the following are true:

- A supported scoresheet can be processed end to end without undocumented manual steps.
- The structured result represents all core game fields listed above or marks unavailable/uncertain fields explicitly.
- Automatically accepted recognition decisions have an accepted error rate of at most 1% on the defined held-out real evaluation set.
- Evaluation reports automation rate separately from accepted error rate; increasing automation must not bypass the error gate.
- Evaluation splits prevent the same document from appearing in both training and evaluation, and isolate writers when writer identity is available.
- Ambiguous, low-confidence, or internally inconsistent results are routed to review or unresolved status.
- The full automated test suite passes in CI from a clean checkout using documented setup instructions.
- A contributor can reproduce the supported dataset/review/training/evaluation workflow from repository documentation.

The evaluation set, its version, and the exact metrics used for a release must be recorded so a v1.0 quality claim is reproducible rather than anecdotal.

## Milestones

### M1 — Trusted evaluation data

Establish reviewed ground truth and a stable real-data evaluation set with document-level isolation and writer-level isolation when possible.

### M2 — Complete FECABA parser

Finish the end-to-end FECABA path and cover the full core game record in the structured output.

### M3 — Measured reliability

Measure each recognition component and the end-to-end parser. Calibrate acceptance thresholds against the v1.0 quality gates.

### M4 — Real-world generalization

Validate scans/photos and previously unseen writers, then fix failure modes that occur outside the development samples.

### M5 — Efficient human review

Minimize unnecessary review while keeping the accepted-error gate intact, and make reviewed corrections reusable in the dataset workflow.

### M6 — FECABA v1.0

Release a reproducible, documented, tested FECABA parser with versioned evaluation evidence and a stable public output contract.

### After v1.0 — Format expansion

Add another scoresheet format only after the FECABA v1.0 gates are met. New formats should reuse the same validation, confidence, review, and evaluation principles.

## Out of Scope Before v1.0

The following are intentionally outside the core roadmap until the FECABA v1.0 gates are met:

- tournament or league management;
- player/team registration systems;
- analytics dashboards or advanced performance analytics;
- live game tracking or broadcasting;
- a general-purpose mobile application or SaaS product;
- support for arbitrary basketball scoresheet formats;
- eliminating human review at the expense of reliability;
- replacing deterministic validation with unverified generative output.

## Decision Filter

Before adding a feature, dependency, model, or workflow, ask:

> Does this materially improve our ability to turn a supported scoresheet into correct, structured, auditable data, or help us prove that quality?

If the answer is no, the work should normally wait until after the current milestone. If the answer is yes, it should also have a measurable acceptance criterion tied to one of the goals above.

