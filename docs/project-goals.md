# Project Goals

## Mission

Basketball Scoresheet Parser turns supported basketball scoresheets into structured, reliable, and auditable data. The project starts with the FECABA scoresheet format and automates only the decisions that meet an explicit confidence and quality threshold. Ambiguous information must remain visible and be routed to review instead of being silently guessed or corrected.

## Product Principle

Correctness comes before automation rate.

Every extracted value that depends on uncertain visual or handwriting recognition should preserve enough evidence to explain the decision, expose confidence when applicable, and use an explicit status such as accepted, review, or unresolved.

## North-star Metrics

The primary quality metric is **accepted error rate**: among recognition decisions the system accepts automatically, at most 1% may be wrong on the defined held-out real evaluation set.

The project must track these metrics together:

- **accepted error rate** — primary safety/quality gate;
- **automation rate** — share of decisions accepted without human review;
- **global accuracy** — overall recognition correctness, including decisions sent to review;
- **review rate** — share of decisions that require human review;
- **coverage/completeness** — share of required core game fields represented in the structured result.

Automation rate is a secondary optimization metric. It may improve only while the accepted-error gate remains satisfied. No arbitrary automation target is set before M1 establishes a trustworthy real-data baseline.

## Priority Order

When work competes for time, prioritize it in this order:

1. data integrity and prevention of silently wrong accepted results;
2. completeness of the FECABA end-to-end parser;
3. reproducible evaluation and evidence of quality;
4. reduction of unnecessary human review;
5. contributor experience and maintainability;
6. new formats and adjacent product features after FECABA v1.0.

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

**Exit criteria:**

- the evaluation corpus has reviewed ground truth rather than inferred labels;
- the corpus and split definition are versioned or reproducibly identified;
- no document appears in both training and evaluation;
- writer isolation is enforced whenever writer identity is available;
- a repeatable command or documented procedure produces baseline quality metrics from the evaluation set.

### M2 — Complete FECABA parser

Finish the end-to-end FECABA path and cover the full core game record in the structured output.

**Exit criteria:**

- a supported FECABA image or PDF can be processed through the public CLI/API without undocumented manual intervention;
- teams, players, participation, starters, period scoring, scoring events, individual fouls, team fouls, and final scores are represented or explicitly marked unavailable/unresolved;
- contradictions and uncertain fields propagate review/unresolved state instead of being silently normalized into accepted values;
- automated integration coverage exercises the end-to-end result structure on representative fixtures.

### M3 — Measured reliability

Measure each recognition component and the end-to-end parser. Calibrate acceptance thresholds against the v1.0 quality gates.

**Exit criteria:**

- component and end-to-end metrics are generated from the M1 evaluation set;
- acceptance thresholds are derived from evaluation evidence rather than chosen only by intuition;
- automatically accepted recognition decisions meet the <=1% accepted-error gate;
- automation rate, global accuracy, accepted error rate, and review rate are reported together so quality cannot be hidden behind a single metric.

### M4 — Real-world generalization

Validate scans/photos and previously unseen writers, then fix failure modes that occur outside the development samples.

**Exit criteria:**

- evaluation includes supported real-world capture conditions rather than only development crops;
- unseen-writer results are reported separately whenever writer identity is available;
- known failure modes discovered during evaluation are documented and either fixed or explicitly routed to review;
- the M3 accepted-error gate continues to hold on the defined generalization evaluation slice.

### M5 — Efficient human review

Minimize unnecessary review while keeping the accepted-error gate intact, and make reviewed corrections reusable in the dataset workflow.

**Exit criteria:**

- review-required decisions can move through a documented review workflow and return as reusable reviewed data;
- review rate and automation rate are measured release over release;
- threshold or model improvements that reduce review do not violate the accepted-error gate;
- stale, contradictory, or invalid reviewed data is detected rather than silently reused.

### M6 — FECABA v1.0

Release a reproducible, documented, tested FECABA parser with versioned evaluation evidence and a stable public output contract.

**Exit criteria:**

- all v1.0 quality gates and M1-M5 exit criteria are satisfied;
- a clean checkout can install the documented dependencies and pass CI;
- the public structured output/schema and supported input expectations are documented;
- the release records the evaluation-set version/identity and measured quality metrics;
- the release is tagged/versioned and can be reproduced by an external contributor from public documentation.

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

## Goal Governance

This document is the source of truth for product direction through FECABA v1.0.

- New work should map to the current milestone, a v1.0 quality gate, or a documented defect that threatens one of them.
- Work that belongs to a later milestone may be explored only when it does not delay the current milestone or weaken its evidence requirements.
- A new feature must state which goal or exit criterion it advances and how completion will be measured before implementation begins.
- Mission, north-star metrics, v1.0 quality gates, milestone exit criteria, or pre-v1.0 scope boundaries should change only through an explicit documented decision, with the reason recorded in the commit/PR that changes this file.
- Passing tests alone does not justify scope expansion; milestone exit evidence is required before advancing to the next phase.
- The project should not declare a milestone complete because remaining work is difficult or manual. Its listed exit criteria are the definition of completion.

When a proposed change conflicts with this document, the default is to keep the current project goals and defer the conflicting work until the goals are deliberately revised.

